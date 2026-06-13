"""gsm8k_newmethod.py — does the VALIDATED recall stack actually let the model SOLVE?

GSM8K task accuracy with the new components wired into generation: the Dolphin-trained
per-position GATE + the learned retriever (QK indexer or state-free BGE head), fire-once
and pin. Each problem's numbers are pushed past the 512-window (so OFF cannot solve);
the gate must fire mid-CoT and pull the right block back for the model to answer.

Arms (greedy, exact-match on the final integer):
  OFF       no recall (numbers gone)            -> floor
  TURN      retrieve the problem block at turn start (question-position q) + pin
  GATE_QK   per-position gate -> learned QK indexer -> inject top block once + pin
  GATE_BGE  per-position gate -> state-free BGE head -> inject top block once + pin
"""
import os, re, json, time
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
try:
    from transformers import DynamicCache
except Exception:
    from transformers.cache_utils import DynamicCache
from datasets import load_dataset
from attn_export3_torch import load_pooler
from recall_kit import RecallGate, QKIndexer, BGERetriever
from recall_kit.runtime import _Cap

DEV = "cuda" if torch.cuda.is_available() else "cpu"
RW, C, CAP, BLOCK = 512, 64, 400, 128
LAYERS = (8, 14, 20)
N = int(os.environ.get("GSM_N", "30"))
ARMS = os.environ.get("ARMS", "OFF,TURN,GATE_QK,GATE_BGE").split(",")
GATE_THRESH = os.environ.get("GATE_THRESH")          # optional override
OUT = os.environ.get("OUT_DIR", "results_gsm_new")
LOG = []
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.append(s)
    os.makedirs(OUT, exist_ok=True); open(os.path.join(OUT, "live.log"), "w").write("\n".join(LOG))

FILLER = ("The quarterly logistics review covered routine warehouse throughput, staff "
          "rotation schedules, the cafeteria menu refresh, parking lot repaving, the "
          "visitor badge policy, fire-drill timing, recycling pickup days, and the lobby "
          "plant rotation. None of these notes changed any quantity or figure. ")
AUG = ("Solve the math problem stated earlier in this conversation. Reason step by step, "
       "then end with 'Final answer: ' followed by the single final number.")
GOLD_RE = re.compile(r"####\s*([-\d,\.]+)")
NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
def norm(s):
    s = s.replace(",", "").strip().rstrip(".")
    try:
        f = float(s); return str(int(f)) if f == int(f) else str(f)
    except Exception:
        return s
def final_num(t):
    m = NUM_RE.findall(t); return norm(m[-1]) if m else ""


def build():
    tok = AutoTokenizer.from_pretrained("fft_hf")
    try:
        llm = AutoModelForCausalLM.from_pretrained("fft_hf", torch_dtype=torch.float32)
    except TypeError:
        llm = AutoModelForCausalLM.from_pretrained("fft_hf", dtype=torch.float32)
    llm = llm.eval().to(DEV)
    pooler = load_pooler().to(DEV).eval()
    from sentence_transformers import SentenceTransformer
    bge = SentenceTransformer("BAAI/bge-base-en-v1.5", device=DEV)
    cfg = llm.config; dims = (cfg.num_attention_heads, cfg.num_key_value_heads,
                              cfg.hidden_size // cfg.num_attention_heads)
    gate = RecallGate.load("art/gate.npz")
    if GATE_THRESH:
        gate.thresh = float(GATE_THRESH)
    qk = QKIndexer.load("art/indexer.npz", dims, device=DEV)
    bge_ret = BGERetriever.load("art/bge_head.npz", bge, device=DEV)
    log(f"[load] dims={dims} gate.thresh={gate.thresh:.3f}")
    return tok, llm, pooler, gate, qk, bge_ret, dims


def emb(embT, ids):
    if ids:
        return embT(torch.tensor([ids], device=DEV))
    return torch.zeros(1, 0, embT.weight.shape[1], dtype=embT.weight.dtype, device=DEV)


@torch.no_grad()
def block_keys(llm, doc_ids, n_evict, nB, dims):
    Hq, Hkv, D = dims
    with _Cap(llm, LAYERS, q=False, k=True) as cap:
        llm(input_ids=torch.tensor([doc_ids], device=DEV), use_cache=False)
    kf = []
    for li in LAYERS:
        k = cap.k[li][0][:n_evict].view(nB, BLOCK, Hkv, D).float()
        kf.append(torch.cat([k.mean(1), k.amax(1)], -1).cpu())
    return torch.stack(kf, 1).half().float().numpy()      # [nB,L,Hkv,2D]


@torch.no_grad()
def solve(tok, llm, pooler, embT, gate, qk, bge_ret, dims, problem, arm):
    Hq, Hkv, D = dims
    THINK = tok.encode("<think>\n", add_special_tokens=False)
    eos = tok.eos_token_id
    doc = problem + " "
    while len(tok.encode(doc, add_special_tokens=False)) < 1200:
        doc += FILLER
    doc_ids = tok.encode(doc, add_special_tokens=False)
    n_evict = ((len(doc_ids) - RW) // BLOCK) * BLOCK
    nB = n_evict // BLOCK
    if nB < 2:
        return None
    block_ids = [doc_ids[i * BLOCK:(i + 1) * BLOCK] for i in range(nB)]
    kept = doc_ids[:n_evict]; window_ctx = doc_ids[n_evict:]
    need_ret = arm != "OFF"
    keys = block_keys(llm, doc_ids, n_evict, nB, dims) if need_ret in (True,) and arm in ("TURN", "GATE_QK") else None
    block_texts = [tok.decode(b) for b in block_ids] if arm in ("GATE_BGE",) else None
    sp = pooler(emb(embT, kept).float()).to(embT.weight.dtype)

    feed = tok.encode(f"<｜end▁of▁sentence｜><｜User｜>{AUG}<｜Assistant｜>",
                      add_special_tokens=False) + list(THINK)
    gen_new, fi, new = [], 0, 0
    rec_emb = None; pinned = False; n_fire = 0; first_fire = None

    # frontier pre-RoPE q capture for the gate + retriever (GATE arms)
    fq = {}; hooks = []
    if arm in ("GATE_QK", "GATE_BGE"):
        for li in LAYERS:
            def mk(li):
                def fn(_m, _i, o):
                    fq[li] = o.detach()[0, -1].float()
                return fn
            hooks.append(llm.model.layers[li].self_attn.q_proj.register_forward_hook(mk(li)))

    cache = DynamicCache()
    bos = tok.bos_token_id
    llm(input_ids=torch.tensor([[bos]], device=DEV), past_key_values=cache, use_cache=True)
    MQ = cache.get_seq_length()

    # TURN: retrieve the problem block once at turn start via the question-position q
    if arm == "TURN":
        c2 = DynamicCache()
        llm.model(input_ids=torch.tensor([[bos]], device=DEV), past_key_values=c2, use_cache=True)
        llm.model(inputs_embeds=torch.cat([sp, emb(embT, window_ctx)], 1), past_key_values=c2, use_cache=True)
        qids = tok.encode(f"<｜User｜>{AUG}<｜Assistant｜>", add_special_tokens=False)
        with _Cap(llm, LAYERS, q=True, k=False) as cap:
            llm.model(inputs_embeds=emb(embT, qids), past_key_values=c2, use_cache=True)
        qf = np.stack([cap.q[li][0].float().cpu().numpy().mean(0).reshape(Hq, D) for li in LAYERS])
        tops = sorted(qk.rank(qf, keys)[:2])
        rec_emb = emb(embT, [t for b in tops for t in block_ids[b]]); pinned = True

    def frontier():
        return np.stack([fq[li].view(Hq, D).cpu().numpy() for li in LAYERS])   # [L,Hq,D]

    done = False
    while not done:
        # gate + retrieve (GATE arms)
        if arm in ("GATE_QK", "GATE_BGE") and not pinned and fq:
            qf = frontier()
            if gate.fires(qf.reshape(-1)):
                n_fire += 1; first_fire = len(gen_new)
                order = qk.rank(qf, keys) if arm == "GATE_QK" else bge_ret.rank(qf.reshape(-1), block_texts)
                tops = sorted(order[:2])
                rec_emb = emb(embT, [t for b in tops for t in block_ids[b]]); pinned = True
        win = (window_ctx + gen_new)[-RW:]
        parts = [sp] + ([rec_emb] if rec_emb is not None else []) + [emb(embT, win)]
        cache.crop(MQ)
        last = llm(inputs_embeds=torch.cat(parts, 1), past_key_values=cache,
                   use_cache=True).logits[:, -1, :].float()
        for _ in range(C):
            if fi < len(feed):
                t = feed[fi]; fi += 1
            else:
                t = int(last[0].argmax())
                if t == eos:
                    done = True; break
                new += 1
                if new >= CAP:
                    done = True; break
                gen_new.append(t)
            last = llm(inputs_embeds=emb(embT, [t]), past_key_values=cache,
                       use_cache=True).logits[:, -1, :].float()
        if done:
            break
    for h in hooks:
        h.remove()
    body = tok.decode(gen_new)
    ans = body.split("</think>")[-1] if "</think>" in body else body
    if "Final answer:" in ans:
        ans = ans.split("Final answer:")[-1]
    return {"pred": final_num(ans), "n_fire": n_fire, "first_fire": first_fire,
            "tokens": len(gen_new)}


def main():
    t0 = time.time(); os.makedirs(OUT, exist_ok=True)
    ds = load_dataset("gsm8k", "main", split="test")
    probs = []
    for i in range(N):
        g = GOLD_RE.search(ds[i]["answer"])
        probs.append({"q": ds[i]["question"], "gold": norm(g.group(1)) if g else ""})
    tok, llm, pooler, gate, qk, bge_ret, dims = build()
    embT = llm.get_input_embeddings()
    agg = {a: 0 for a in ARMS}; results = []; n = 0
    for i, p in enumerate(probs):
        rec = {"i": i, "gold": p["gold"], "arms": {}}
        line = f"[{i:2}] gold={p['gold']:>6} "
        ok = True
        for a in ARMS:
            r = solve(tok, llm, pooler, embT, gate, qk, bge_ret, dims, p["q"], a)
            if r is None:
                ok = False; break
            c = r["pred"] == p["gold"] and p["gold"] != ""
            agg[a] += int(c); rec["arms"][a] = {**r, "correct": c}
            line += f"| {a}={'OK' if c else 'x '}({r['pred']:>6},f{r['n_fire']}) "
        if not ok:
            continue
        n += 1; results.append(rec); log(line + f" [{n}/{N} t={time.time()-t0:.0f}s]")
        payload = {"n": n, "running": agg, "results": results, "elapsed_s": round(time.time()-t0,1)}
        json.dump(payload, open(os.path.join(OUT, "results.json"), "w"), indent=1)
    summary = {a: f"{agg[a]}/{n} = {agg[a]/max(n,1):.1%}" for a in ARMS}
    log("\n=== GSM8K (numbers SP-evicted; new method; greedy) ===")
    for a in ARMS:
        log(f"  {a:9}: {summary[a]}")
    payload = {"n": n, "summary": summary, "running": agg, "results": results,
               "elapsed_s": round(time.time()-t0,1), "final": True}
    json.dump(payload, open(os.path.join(OUT, "results.json"), "w"), indent=1)
    open(os.path.join(OUT, ".done"), "w").write("ok")
    log(f"DONE t={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
