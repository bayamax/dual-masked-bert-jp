"""cotscan_task_eval.py — TASK eval of mid-CoT recall with the CORRECT components.

For each held-out natural multi-turn conversation (needle value evicted past the window):
the model must answer an OBLIQUE compute question that needs the evicted value mid-CoT.
Three arms, the ONLY difference being how/when the evicted block is re-injected:

  OFF  : SP gist + window only (value is gone).
  TURN : retrieve once at turn start with the question-position query via the LEARNED
         Phase B indexer; inject the top block; pin.
  GATE : per-position recall gate (SP-context probe) fires when the frontier hidden state
         crosses threshold; THEN retrieve via the indexer with that frontier query, inject
         the top block ONCE and pin (no per-step refresh).

Metrics: value_recovered = the evicted value string appears in the generated output;
         task_correct     = value*mult (the intended answer) appears.
"""
import os, sys, json, re, time
import numpy as np
import torch, torch.nn.functional as F
try:
    from transformers import DynamicCache
except Exception:
    from transformers.cache_utils import DynamicCache
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transformers import AutoModelForCausalLM, AutoTokenizer
from attn_export3_torch import load_pooler
from cot_recall_eval import Indexer
from cot_recall_labels import LAYERS, WINDOW, BLOCK, build_conversation, to_ids, ITEMS
from recall_query_labels import Capture

DEV = "cuda" if torch.cuda.is_available() else "cpu"
RW, C, CAP = 512, 64, 400
SEEDS = int(os.environ.get("TASK_SEEDS", "3"))

LOG = []
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.append(s)


def load_components(path="task_components.npz"):
    z = np.load(path)
    L, Hq, Hkv, D = int(z["L"]), int(z["Hq"]), int(z["Hkv"]), int(z["D"])
    ix = Indexer(L, Hq, Hkv, D)
    ix.load_state_dict({"Aq": torch.tensor(z["ix_Aq"]), "Bk": torch.tensor(z["ix_Bk"]),
                        "w": torch.tensor(z["ix_w"])})
    ix.eval()
    gate = {"coef": z["g_coef"].astype(np.float32), "b": float(z["g_intercept"]),
            "mean": z["g_mean"].astype(np.float32), "scale": z["g_scale"].astype(np.float32),
            "thresh": float(os.environ.get("GATE_THRESH", z["g_thresh"]))}
    log(f"[components] gate AUC~{float(z['g_auc']):.3f} thresh={gate['thresh']:.3f} "
        f"indexer heldout_top2={float(z['ho_top2']):.3f} dims L{L} Hq{Hq} Hkv{Hkv} D{D}")
    return gate, ix, (L, Hq, Hkv, D)


def emb(embT, ids):
    if ids:
        return embT(torch.tensor([ids], device=DEV))
    return torch.zeros(1, 0, embT.weight.shape[1], dtype=embT.weight.dtype, device=DEV)


@torch.no_grad()
def context_block_keys(llm, ids, n_evict, nB, dims):
    """Contextual pre-RoPE block keys [nB,L,Hkv,2D] from one full-KV prefill (matches the
    indexer's training-time key harvest)."""
    L, Hq, Hkv, D = dims
    cache = DynamicCache()
    with Capture(llm, LAYERS, want_q=False, want_k=True) as cap:
        llm.model(input_ids=torch.tensor([ids], device=DEV), past_key_values=cache,
                  use_cache=True)
    kf = []
    for li in LAYERS:
        k = cap.k[li][0][0][:n_evict].view(nB, BLOCK, Hkv, D).float()
        kf.append(torch.cat([k.mean(1), k.amax(1)], dim=-1).cpu())
    return torch.stack(kf, dim=1)                       # [nB,L,Hkv,2D]


def indexer_top_block(ix, q_LHqD, keys, blocks_ids):
    """Return the token ids of the indexer's top-1 block."""
    with torch.no_grad():
        s = ix(q_LHqD.float().cpu(), keys.float())
    return blocks_ids[int(s.argmax())]


@torch.no_grad()
def run_arm(llm, tok, pooler, embT, ids, q_text, value, mult, gate, ix, dims, arm):
    L, Hq, Hkv, D = dims
    THINK = tok.encode("<think>\n", add_special_tokens=False)
    eos = tok.eos_token_id
    n_evict = ((len(ids) - WINDOW) // BLOCK) * BLOCK
    nB = n_evict // BLOCK
    if nB < 4:
        return None
    blocks_ids = [ids[i * BLOCK:(i + 1) * BLOCK] for i in range(nB)]
    kept = ids[:n_evict]; window_ctx = ids[n_evict:]
    keys = context_block_keys(llm, ids, n_evict, nB, dims) if arm != "OFF" else None
    sp = pooler(emb(embT, kept).float()).to(embT.weight.dtype)          # constant gist

    feed = tok.encode(f"<｜end▁of▁sentence｜><｜User｜>{q_text}<｜Assistant｜>",
                      add_special_tokens=False) + list(THINK)
    gen_new, fi, new = [], 0, 0
    rec_emb = None; pinned = False; fired_at = None; gate_scores = []

    # frontier pre-RoPE q capture (single latest position), for gate + indexer query
    fq = {}
    hooks = []
    if arm == "GATE":
        for li in LAYERS:
            def mk(li):
                def fn(_m, _i, o):
                    fq[li] = o.detach()[0, -1].float()                 # [Hq*D]
                return fn
            hooks.append(llm.model.layers[li].self_attn.q_proj.register_forward_hook(mk(li)))

    cache = DynamicCache()
    bos = tok.bos_token_id
    llm(input_ids=torch.tensor([[bos]], device=DEV), past_key_values=cache, use_cache=True)
    MQ = cache.get_seq_length()

    # TURN: question-position query -> indexer -> inject top block, pinned from the start
    if arm == "TURN":
        c2 = DynamicCache()
        llm.model(input_ids=torch.tensor([[bos]], device=DEV), past_key_values=c2, use_cache=True)
        llm.model(inputs_embeds=torch.cat([sp, emb(embT, window_ctx)], 1),
                  past_key_values=c2, use_cache=True)
        q_ids = tok.encode(f"<｜User｜>{q_text}<｜Assistant｜>", add_special_tokens=False)
        with Capture(llm, LAYERS, want_q=True, want_k=False) as cap:
            llm.model(inputs_embeds=emb(embT, q_ids), past_key_values=c2, use_cache=True)
        q = torch.stack([cap.q[li][0][0].mean(0).view(Hq, D) for li in LAYERS])  # [L,Hq,D]
        rec_emb = emb(embT, indexer_top_block(ix, q, keys, blocks_ids)); pinned = True

    def gate_score():
        if any(li not in fq for li in LAYERS):
            return None
        f = torch.cat([fq[li] for li in LAYERS]).cpu().numpy()
        return float(((f - gate["mean"]) / gate["scale"]) @ gate["coef"] + gate["b"])

    def gate_query():
        return torch.stack([fq[li].view(Hq, D) for li in LAYERS])               # [L,Hq,D]

    done = False
    while not done:
        if arm == "GATE" and not pinned and fq:
            s = gate_score()
            if s is not None:
                gate_scores.append(round(s, 2))
                if s > gate["thresh"]:
                    rec_emb = emb(embT, indexer_top_block(ix, gate_query(), keys, blocks_ids))
                    pinned = True; fired_at = len(gen_new)
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
    nv = value.replace(",", "")
    try:
        gold = float(value) * mult
        gold_s = str(int(gold)) if gold == int(gold) else str(gold)
    except Exception:
        gold_s = None
    return {"arm": arm, "value_recovered": nv in body.replace(",", ""),
            "task_correct": bool(gold_s and gold_s in body.replace(",", "")),
            "fired_at": fired_at, "n_gate_checks": len(gate_scores),
            "gate_max": max(gate_scores) if gate_scores else None,
            "answer": body[-200:]}


def main():
    t0 = time.time()
    out_dir = os.environ.get("OUT_DIR", "results_taskeval")
    os.makedirs(out_dir, exist_ok=True)
    gate, ix, dims = load_components()
    tok = AutoTokenizer.from_pretrained("fft_hf")
    try:
        llm = AutoModelForCausalLM.from_pretrained("fft_hf", dtype=torch.float32)
    except TypeError:
        llm = AutoModelForCausalLM.from_pretrained("fft_hf", torch_dtype=torch.float32)
    llm = llm.eval().to(DEV)
    pooler = load_pooler().to(DEV).eval()
    embT = llm.get_input_embeddings()

    split = len(ITEMS) * 3 // 4
    held_items = list(range(split, len(ITEMS)))
    arms = ["OFF", "TURN", "GATE"]
    agg = {a: {"val": 0, "task": 0} for a in arms}
    results = []; n = 0
    for s in range(SEEDS):
        for item_idx in held_items:
            seed = 700000 + s * 1000 + item_idx
            built = build_conversation(seed, item_idx)
            if built is None:
                continue
            pairs, needle_user, q_text, lead, value = built
            ids, _ = to_ids(tok, pairs)
            mult = ITEMS[item_idx].get("mult", 1)
            rec = {"seed": seed, "item": item_idx, "value": value, "mult": mult, "arms": {}}
            ok = True
            for arm in arms:
                r = run_arm(llm, tok, pooler, embT, ids, q_text, value, mult, gate, ix, dims, arm)
                if r is None:
                    ok = False; break
                rec["arms"][arm] = r
                agg[arm]["val"] += int(r["value_recovered"]); agg[arm]["task"] += int(r["task_correct"])
            if not ok:
                continue
            n += 1; results.append(rec)
            line = " | ".join(f"{a}:val={int(rec['arms'][a]['value_recovered'])}"
                              f",task={int(rec['arms'][a]['task_correct'])}" for a in arms)
            g = rec["arms"]["GATE"]
            log(f"[{n:2}] item{item_idx} v={value} x{mult}  {line}  gate(@{g['fired_at']})")
            snap(out_dir, results, agg, n, t0)
    summary = {a: {"value_recovered": f"{agg[a]['val']}/{n}", "task_correct": f"{agg[a]['task']}/{n}"}
               for a in arms}
    log("\n=== TASK EVAL (correct components: indexer + gate probe, fire-once-pin) ===")
    for a in arms:
        log(f"  {a:4}: value_recovered {agg[a]['val']}/{n}  task_correct {agg[a]['task']}/{n}")
    snap(out_dir, results, agg, n, t0, summary)
    log(f"DONE n={n} t={time.time()-t0:.0f}s")


def snap(out_dir, results, agg, n, t0, summary=None):
    p = {"n": n, "elapsed_s": round(time.time() - t0, 1), "device": DEV,
         "running": agg, "results": results}
    if summary:
        p["summary"] = summary; p["final"] = True
    json.dump(p, open(os.path.join(out_dir, "results.json"), "w"), ensure_ascii=False, indent=2)
    open(os.path.join(out_dir, "live.log"), "w").write("\n".join(LOG))


if __name__ == "__main__":
    main()
