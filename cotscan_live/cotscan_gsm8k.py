"""cotscan_gsm8k.py — COTSCAN mid-CoT recall on GSM8K (larger, multi-step battery).

Same machinery as cotscan_live.py, but a real benchmark: each GSM8K problem statement
(its numbers included) is pushed OUT of the rw=512 exposure window by filler, so the
model only sees [SP-gist][window][ "solve the problem above, step by step" ]. The numbers
needed for the multi-step solution live in the evicted archive. This is the deferred /
multi-step regime the spec predicted MID should help with.

Arms (only the recall-query source differs):
  OFF  : no recall            -> numbers are gone; floor.
  TURN : recall once at turn start, query = the instruction (current production).
  MID  : recall every rebuild, query = the current CoT tail (COTSCAN per-position).

Decoding is GREEDY (deterministic) so arm differences are attributable to recall, not RNG.
Scoring = exact match of the final integer against the GSM8K gold answer.
"""
import os, re, json, time
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
try:
    from transformers import DynamicCache
except Exception:
    from transformers.cache_utils import DynamicCache
from datasets import load_dataset

from block_recall import BlockArchive
from attn_export3_torch import load_pooler

DEV = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DIR = os.environ.get("FFT_HF", "fft_hf")
RW, C, CAP, RECALL_K = 512, 64, 512, 2
LAYERS = (8, 14, 20)
N = int(os.environ.get("GSM8K_N", "25"))
ARMS = os.environ.get("ARMS", "OFF,TURN,MID").split(",")

LOG = []
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.append(s)

FILLER = (
    "The quarterly logistics review covered routine warehouse throughput, staff rotation "
    "schedules, the cafeteria menu refresh, parking lot repaving, the visitor badge policy, "
    "fire-drill timing, recycling pickup days, the lobby plant rotation, and printer toner "
    "restocking. None of these administrative notes changed any quantities or figures. "
)

GOLD_RE = re.compile(r"####\s*([-\d,\.]+)")
NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")

def norm_num(s):
    s = s.replace(",", "").strip().rstrip(".")
    try:
        f = float(s); return str(int(f)) if f == int(f) else str(f)
    except Exception:
        return s

def final_number(text):
    nums = NUM_RE.findall(text)
    return norm_num(nums[-1]) if nums else ""


def build_model():
    log(f"[load] device={DEV} model={MODEL_DIR}")
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    try:
        llm = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float32)
    except TypeError:
        llm = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=torch.float32)
    llm = llm.eval().to(DEV)
    pooler = load_pooler().to(DEV).eval()
    return tok, llm, pooler, llm.get_input_embeddings()


def emb(embT, ids):
    if ids:
        return embT(torch.tensor([ids], device=DEV))
    return torch.zeros(1, 0, embT.weight.shape[1], dtype=embT.weight.dtype, device=DEV)


@torch.no_grad()
def solve(tok, llm, pooler, embT, problem, arm, gate=None):
    """Greedy CoT solve where `problem` is buried in the evicted archive."""
    THINK_OPEN = tok.encode("<think>\n", add_special_tokens=False)
    eos = tok.eos_token_id
    doc = problem + " " + (FILLER * 10)        # bury the numbers past rw=512
    aug = ("Solve the math problem stated earlier in this conversation. Reason step by "
           "step, then end with 'Final answer: ' followed by the single final number.")

    gen = tok.encode(doc, add_special_tokens=False)
    archive = BlockArchive(llm, tok, layers=LAYERS, block=128, mode="qk")
    R0 = min(len(gen), RW); nd_end = len(gen) - R0
    archive.extend(gen[:nd_end]); kept = list(gen[:nd_end]); absorbed = nd_end

    feed = tok.encode(f"<｜User｜>{aug}<｜Assistant｜>", add_special_tokens=False) + list(THINK_OPEN)
    turn_start = len(gen)
    cache = DynamicCache()
    bos = tok.bos_token_id if tok.bos_token_id is not None else eos
    prime_last = llm(input_ids=torch.tensor([[bos]], device=DEV),
                     past_key_values=cache, use_cache=True).logits[:, -1, :].float()
    MQ = cache.get_seq_length()

    rec_emb = None; fi = new = 0; in_think = True; done = False; n_recall = 0
    acc_score = {}                                  # MID_ACC: cumulative block scores
    pinned = False; gate_scores = []; first_fire = None  # GATE arm bookkeeping

    def qids():
        tail = gen[turn_start:]
        return tail[-48:] if len(tail) >= 4 else tok.encode(aug, add_special_tokens=False)

    # GATE arm: per-position recall gate (TRIGGER_V3 head applied IN the generation loop,
    # COTSCAN残作業#1). Hook q_proj at layers 8/14/20, keep the latest position's pre-RoPE
    # query; the gate fires only when that frontier query crosses the head's threshold —
    # then we inject the problem block ONCE and PIN it (no per-step refresh, per the GSM8K
    # v3 finding that re-injection itself destabilizes).
    gq = {}; hooks = []
    if arm == "GATE" and gate is not None:
        for li in LAYERS:
            def mk(li):
                def fn(_m, _i, o):
                    gq[li] = o.detach()[0, -1].float().cpu().numpy().ravel()
                return fn
            hooks.append(llm.model.layers[li].self_attn.q_proj.register_forward_hook(mk(li)))

    def gate_score():
        if any(li not in gq for li in LAYERS):
            return None
        f = np.concatenate([gq[li] for li in LAYERS])
        return float(((f - gate["mean"]) / gate["scale"]) @ gate["coef"] + gate["b"])

    while not done:
        c0 = len(gen); R = min(c0, RW); nd = c0 - R
        if nd > absorbed:
            archive.extend(gen[absorbed:nd]); kept.extend(gen[absorbed:nd]); absorbed = nd
        if arm == "OFF":
            rec_emb = None
        elif arm == "TURN":
            if rec_emb is None and (archive.blocks or len(archive.buf) >= 16):
                rid = archive.retrieve(tok.encode(aug, add_special_tokens=False), k=RECALL_K)
                if rid:
                    rec_emb = emb(embT, rid); n_recall += 1
        elif arm == "MID":
            if archive.blocks or len(archive.buf) >= 16:
                rid = archive.retrieve(qids(), k=RECALL_K)
                if rid:
                    rec_emb = emb(embT, rid); n_recall += 1
        elif arm == "MID_Q":
            # ablation: re-inject EVERY rebuild (like MID) but keep the GOOD query (the
            # instruction, like TURN). Isolates "frequent re-injection" from "CoT-tail query".
            if archive.blocks or len(archive.buf) >= 16:
                rid = archive.retrieve(tok.encode(aug, add_special_tokens=False), k=RECALL_K)
                if rid:
                    rec_emb = emb(embT, rid); n_recall += 1
        elif arm == "MID_ACC":
            # per-position but ACCUMULATE: each rebuild adds the current tail's top-k block
            # scores; the injected context is the union of the top blocks by cumulative
            # score, so a block that keeps matching (the problem statement) stays PINNED
            # instead of being swapped out the moment the CoT tail drifts.
            if archive.blocks or len(archive.buf) >= 16:
                archive.retrieve(qids(), k=RECALL_K)
                sc = getattr(archive, "_last_scores", []) or []
                nb = len(archive.blocks)
                for idx in sorted(range(len(sc)), key=lambda i: -sc[i])[:RECALL_K]:
                    if idx < nb:                       # ignore the volatile pending buffer
                        acc_score[idx] = acc_score.get(idx, 0.0) + sc[idx]
                if acc_score:
                    keep = sorted(sorted(acc_score, key=lambda i: -acc_score[i])[:3])
                    ids = []
                    for i in keep:
                        ids += archive.blocks[i]["ids"]
                    rec_emb = emb(embT, ids); n_recall += 1
        elif arm == "GATE":
            # fire only when the frontier hidden state says "I need evicted info"; then
            # inject the problem block once (instruction query = the reliable key) and pin.
            if not pinned and (archive.blocks or len(archive.buf) >= 16):
                s = gate_score()
                if s is not None:
                    gate_scores.append(round(s, 2))
                    if s > gate["thresh"]:
                        rid = archive.retrieve(tok.encode(aug, add_special_tokens=False),
                                               k=RECALL_K)
                        if rid:
                            rec_emb = emb(embT, rid); n_recall += 1
                            pinned = True; first_fire = len(gen) - turn_start

        parts = []
        if kept:
            parts.append(pooler(emb(embT, kept).float()).to(embT.weight.dtype))
        if rec_emb is not None:
            parts.append(rec_emb.to(embT.weight.dtype))
        if R > 0:
            parts.append(emb(embT, gen[c0 - R:c0]))
        cache.crop(MQ)
        if parts:
            last = llm(inputs_embeds=torch.cat(parts, 1), past_key_values=cache,
                       use_cache=True).logits[:, -1, :].float()
        else:
            last = prime_last
        for _ in range(C):
            if fi < len(feed):
                t = feed[fi]; fi += 1
            else:
                t = int(last[0].argmax())          # greedy
                if t == eos:
                    done = True; break
                new += 1
                if new >= CAP:
                    done = True; break
            gen.append(t)
            if in_think and "</think>" in tok.decode(gen[-8:]):
                in_think = False
            last = llm(inputs_embeds=emb(embT, [t]), past_key_values=cache,
                       use_cache=True).logits[:, -1, :].float()
        if done:
            break

    body = tok.decode(gen[turn_start:]).split("<｜Assistant｜>", 1)[-1]
    # close an unterminated think with a short greedy salvage so we get a final number
    if "</think>" not in body:
        for t in tok.encode("\n</think>\n\nFinal answer: ", add_special_tokens=False):
            gen.append(t)
            last = llm(inputs_embeds=emb(embT, [t]), past_key_values=cache,
                       use_cache=True).logits[:, -1, :].float()
        for _ in range(24):
            t = int(last[0].argmax())
            if t == eos:
                break
            gen.append(t)
            last = llm(inputs_embeds=emb(embT, [t]), past_key_values=cache,
                       use_cache=True).logits[:, -1, :].float()
        body = tok.decode(gen[turn_start:]).split("<｜Assistant｜>", 1)[-1]
    for h in hooks:
        h.remove()
    ans = body.split("</think>")[-1].strip()
    if "Final answer:" in ans:
        ans = ans.split("Final answer:")[-1]
    out = {"pred": final_number(ans), "n_recall": n_recall,
           "tokens": len(gen) - turn_start, "answer": ans[:200]}
    if arm == "GATE":
        out.update({"fired": pinned, "first_fire_tok": first_fire,
                    "gate_max": max(gate_scores) if gate_scores else None,
                    "gate_thresh": gate["thresh"], "n_gate_checks": len(gate_scores)})
    return out


def main():
    t0 = time.time()
    out_dir = os.environ.get("OUT_DIR", "results_gsm8k")
    os.makedirs(out_dir, exist_ok=True)
    log(f"[data] loading gsm8k test, N={N}")
    ds = load_dataset("gsm8k", "main", split="test")
    probs = []
    for i in range(N):
        ex = ds[i]
        g = GOLD_RE.search(ex["answer"])
        probs.append({"q": ex["question"], "gold": norm_num(g.group(1)) if g else ""})

    tok, llm, pooler, embT = build_model()
    gate = None
    if "GATE" in ARMS:
        gp = os.environ.get("GATE_HEAD", "gate/trigger_head_v3_gate.npz")
        z = np.load(gp)
        gate = {"coef": z["coef"].astype(np.float32).ravel(),
                "b": float(np.ravel(z["intercept"])[0]),
                "mean": z["mean"].astype(np.float32), "scale": z["scale"].astype(np.float32),
                "thresh": float(os.environ.get("GATE_THRESH", z["thresh"])
                                if "thresh" in z.files else os.environ.get("GATE_THRESH", 0.0))}
        log(f"[gate] loaded {gp} thresh={gate['thresh']} dim={gate['coef'].size}")
    results = []
    agg = {a: 0 for a in ARMS}
    for i, p in enumerate(probs):
        rec = {"i": i, "gold": p["gold"], "arms": {}}
        line = f"[{i:2}] gold={p['gold']:>6} "
        for arm in ARMS:
            r = solve(tok, llm, pooler, embT, p["q"], arm, gate=gate)
            ok = r["pred"] == p["gold"] and p["gold"] != ""
            agg[arm] += int(ok)
            rec["arms"][arm] = {**r, "correct": ok}
            line += f"| {arm}={'OK ' if ok else 'x  '}({r['pred']:>6}) "
        results.append(rec)
        log(line + f"  [{i+1}/{N} t={time.time()-t0:.0f}s]")
        snapshot(out_dir, results, agg, t0, i + 1)
    summary = {a: f"{agg[a]}/{N} = {agg[a]/N:.1%}" for a in ARMS}
    log("\n=== GSM8K ACCURACY (numbers SP-evicted; greedy) ===")
    for a in ARMS:
        log(f"  {a:4}: {summary[a]}")
    snapshot(out_dir, results, agg, t0, N, summary=summary, final=True)
    log(f"DONE t={time.time()-t0:.0f}s")


def snapshot(out_dir, results, agg, t0, done, summary=None, final=False):
    payload = {"elapsed_s": round(time.time() - t0, 1), "device": DEV, "N": N,
               "done": done, "decode": "greedy", "running_acc": dict(agg),
               "config": {"rw": RW, "C": C, "cap": CAP, "recall_k": RECALL_K, "layers": LAYERS},
               "results": results}
    if summary:
        payload["summary"] = summary
    if final:
        payload["final"] = True
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "live.log"), "w") as f:
        f.write("\n".join(LOG))


if __name__ == "__main__":
    main()
