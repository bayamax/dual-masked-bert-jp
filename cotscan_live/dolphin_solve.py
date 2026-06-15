"""dolphin_solve.py — the long-context regime where SP compression actually bites.

Dolphin-R1 queries produce LONG CoTs (>rw), so the early reasoning is genuinely absorbed
into the SP gist mid-generation (unlike GSM8K whose CoT fits the window). Three arms on the
SAME fft_hf model, greedy:
  FULL        full KV, no compression (reference ceiling)
  SP          SP gist + rw window (compression kicks in when CoT>rw)
  SP_RECALL   SP + raw-QK archive of the model's own evicted reasoning; the Dolphin gate
              fires when the frontier attends past the window, then top-2 evicted blocks
              are re-injected (so the model can recover its own scrolled-out steps).

Score: exact-match of the final number vs the Dolphin reference answer, AND agreement with
the FULL arm (how far SP drifts from uncompressed, and how much RECALL closes that gap).
"""
import os, re, json, time
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
try:
    from transformers import DynamicCache
except Exception:
    from transformers.cache_utils import DynamicCache
from datasets import load_dataset
from attn_export3_torch import load_pooler
from block_recall import BlockArchive

DEV = "cuda" if torch.cuda.is_available() else "cpu"
RW = int(os.environ.get("RW", "1024"))
C, CAP, BLOCK, RECALL_K = 64, int(os.environ.get("CAP", "1500")), 128, 2
LAYERS = (8, 14, 20)
N = int(os.environ.get("DOLPHIN_N", "20"))
ARMS = os.environ.get("ARMS", "FULL,SP,SP_RECALL").split(",")
GATE_THRESH = float(os.environ.get("GATE_THRESH", "0"))
OUT = os.environ.get("OUT_DIR", "results_dolphin_solve")
LOG = []
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.append(s)
    os.makedirs(OUT, exist_ok=True); open(os.path.join(OUT, "live.log"), "w").write("\n".join(LOG))

NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
BOX_RE = re.compile(r"\\boxed\{([^}]*)\}")
def norm(s):
    s = s.replace(",", "").replace("$", "").strip().rstrip(".")
    try:
        f = float(s); return str(int(f)) if f == int(f) else str(f)
    except Exception:
        return s
def final_num(t):
    b = BOX_RE.findall(t)
    if b:
        m = NUM_RE.findall(b[-1])
        if m:
            return norm(m[-1])
    m = NUM_RE.findall(t)
    return norm(m[-1]) if m else ""


def emb(embT, ids):
    if ids:
        return embT(torch.tensor([ids], device=DEV))
    return torch.zeros(1, 0, embT.weight.shape[1], dtype=embT.weight.dtype, device=DEV)


@torch.no_grad()
def solve(tok, llm, pooler, embT, gate, problem, arm):
    eos = tok.eos_token_id
    THINK = tok.encode("<think>\n", add_special_tokens=False)
    instr = " Reason step by step, then give the final answer in \\boxed{}."
    feed = tok.encode(f"<｜User｜>{problem}{instr}<｜Assistant｜>", add_special_tokens=False) + list(THINK)
    gen, kept, absorbed = [], [], 0
    fi = new = 0; done = False; max_kept = 0; n_fire = 0
    archive = BlockArchive(llm, tok, layers=LAYERS, block=BLOCK, mode="qk") if arm == "SP_RECALL" else None
    rec_emb = None; trace = []
    gq = {}; hooks = []
    if arm == "SP_RECALL":
        for li in LAYERS:
            def mk(li):
                def fn(_m, _i, o):
                    gq[li] = o.detach()[0, -1].float().cpu().numpy().ravel()
                return fn
            hooks.append(llm.model.layers[li].self_attn.q_proj.register_forward_hook(mk(li)))

    def gscore():
        if gate is None or any(li not in gq for li in LAYERS):
            return None
        f = np.concatenate([gq[li] for li in LAYERS])
        return float(((f - gate["mean"]) / gate["scale"]) @ gate["coef"] + gate["b"])

    cache = DynamicCache()
    bos = tok.bos_token_id if tok.bos_token_id is not None else eos
    prime = llm(input_ids=torch.tensor([[bos]], device=DEV), past_key_values=cache, use_cache=True)
    prime_last = prime.logits[:, -1, :].float(); MQ = cache.get_seq_length()
    full = (arm == "FULL")
    while not done:
        c0 = len(gen); R = c0 if full else min(c0, RW); nd = c0 - R
        if nd > absorbed:
            if archive is not None:
                archive.extend(gen[absorbed:nd])
            kept.extend(gen[absorbed:nd]); absorbed = nd; max_kept = max(max_kept, len(kept))
        # SP_RECALL: gate decides whether to (re)pull the model's evicted reasoning
        if arm == "SP_RECALL" and (archive.blocks or len(archive.buf) >= 16):
            s = gscore()
            if s is not None and s > GATE_THRESH:
                tail = gen[-48:] if len(gen) >= 4 else feed
                rid = archive.retrieve(tail, k=RECALL_K)
                if rid:
                    rec_emb = emb(embT, rid); n_fire += 1
                    trace.append({"at_tok": len(gen), "score": round(s,2),
                                  "pulled": tok.decode(rid)[:170],
                                  "cot_here": tok.decode(gen[-26:])})
        parts = []
        if kept and not full:
            parts.append(pooler(emb(embT, kept).float()).to(embT.weight.dtype))
        if rec_emb is not None:
            parts.append(rec_emb.to(embT.weight.dtype))
        if R > 0:
            parts.append(emb(embT, gen[c0 - R:c0]))
        cache.crop(MQ)
        last = (llm(inputs_embeds=torch.cat(parts, 1), past_key_values=cache,
                    use_cache=True).logits[:, -1, :].float() if parts else prime_last)
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
            gen.append(t)
            last = llm(inputs_embeds=emb(embT, [t]), past_key_values=cache,
                       use_cache=True).logits[:, -1, :].float()
        if done:
            break
    for h in hooks:
        h.remove()
    body = tok.decode(gen)
    return final_num(body), {"tokens": len(gen), "kept": max_kept, "n_fire": n_fire,
            "body": body[:1800], "trace": trace}


def pick(tok, n):
    ds = load_dataset("mlabonne/dolphin-r1-deepseek", split="train", streaming=True)
    out = []
    for ex in ds:
        msgs = ex.get("messages") or []
        u = next((m["content"] for m in msgs if m["role"] == "user"), "")
        a = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
        if not u or not a or len(u) > 1500:
            continue
        m = re.search(r"begin_of_thought\|>(.*?)<\|end_of_thought", a, re.S)
        cot = m.group(1) if m else a
        if len(cot) // 4 < 1500:                       # long CoT only (compression will bite)
            continue
        sol = a.split("end_of_thought|>")[-1]
        gold = final_num(sol)
        if not gold:                                   # need a numeric gold to score
            continue
        out.append({"q": u, "gold": gold})
        if len(out) >= n:
            break
    return out


def main():
    t0 = time.time(); os.makedirs(OUT, exist_ok=True)
    tok = AutoTokenizer.from_pretrained("fft_hf")
    try:
        llm = AutoModelForCausalLM.from_pretrained("fft_hf", torch_dtype=torch.float32)
    except TypeError:
        llm = AutoModelForCausalLM.from_pretrained("fft_hf", dtype=torch.float32)
    llm = llm.eval().to(DEV)
    pooler = load_pooler().to(DEV).eval()
    embT = llm.get_input_embeddings()
    gate = None
    if "SP_RECALL" in ARMS and os.path.exists("dolphin_gate.npz"):
        z = np.load("dolphin_gate.npz")
        gate = {"coef": z["coef"].ravel(), "b": float(np.ravel(z["intercept"])[0]),
                "mean": z["mean"], "scale": z["scale"]}
    probs = pick(tok, N)
    log(f"[dolphin] {len(probs)} long-CoT numeric queries | RW={RW} cap={CAP} arms={ARMS}")
    agg = {a: {"correct": 0, "agree_full": 0} for a in ARMS}
    results = []
    for i, p in enumerate(probs):
        rec = {"i": i, "gold": p["gold"], "arms": {}}
        full_ans = None
        for a in ARMS:
            pred, meta = solve(tok, llm, pooler, embT, gate, p["q"], a)
            if a == "FULL":
                full_ans = pred
            ok = pred == p["gold"]
            agg[a]["correct"] += int(ok)
            agg[a]["agree_full"] += int(full_ans is not None and pred == full_ans)
            rec["arms"][a] = {"pred": pred, "correct": ok, **meta}
        results.append(rec)
        line = " | ".join(f"{a}={rec['arms'][a]['pred']}({'OK' if rec['arms'][a]['correct'] else 'x'},k{rec['arms'][a]['kept']}{',f'+str(rec['arms'][a]['n_fire']) if a=='SP_RECALL' else ''})" for a in ARMS)
        log(f"[{i:2}] gold={p['gold']:>6} {line}  [{i+1}/{len(probs)} t={time.time()-t0:.0f}s]")
        n = i + 1
        summary = {a: {"acc": f"{agg[a]['correct']}/{n}", "agree_full": f"{agg[a]['agree_full']}/{n}"} for a in ARMS}
        json.dump({"n": n, "summary": summary, "results": results}, open(os.path.join(OUT, "results.json"), "w"), indent=1)
    n = len(probs)
    summary = {a: {"acc": f"{agg[a]['correct']}/{n} = {agg[a]['correct']/max(n,1):.1%}",
                   "agree_full": f"{agg[a]['agree_full']}/{n} = {agg[a]['agree_full']/max(n,1):.1%}"} for a in ARMS}
    log("\n=== DOLPHIN long-CoT (SP compression regime) ===")
    for a in ARMS:
        log(f"  {a:9}: acc {summary[a]['acc']} | agree-with-FULL {summary[a]['agree_full']}")
    json.dump({"n": n, "summary": summary, "results": results, "final": True}, open(os.path.join(OUT, "results.json"), "w"), indent=1)
    open(os.path.join(OUT, ".done"), "w").write("ok")
    log(f"DONE t={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
