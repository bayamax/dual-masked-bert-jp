"""chat_demo.py — a REAL multi-turn conversation (authored by Claude as the human) driven
through the SP runtime with the NEW gate @ threshold 1.0 + raw-QK recall. Early facts are
stated, then buried under casual turns until they fall out of the 512 window (SP gist only),
then asked back. We run the SAME conversation under three arms and print the full transcript:

  FULL          full KV (facts still in context = upper bound)
  SP            SP gist + window, NO recall (facts evicted = should fail)
  RECALL@1.0    SP + new gate(thresh=1.0) + raw-QK recall of the evicted turns

Each turn's assistant reply is generated greedily, history carried across turns (the runtime
rebuilds the SP gist of evicted turns + window every turn). On the recall-question turns we
check whether the expected fact appears in the reply.
"""
import os, re, json, time
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
try:
    from transformers import DynamicCache
except Exception:
    from transformers.cache_utils import DynamicCache
from attn_export3_torch import load_pooler
from block_recall import BlockArchive

DEV = "cuda" if torch.cuda.is_available() else "cpu"
RW = int(os.environ.get("RW", "512"))
C, BLOCK, RECALL_K = 64, 128, 2
TURN_CAP = int(os.environ.get("TURN_CAP", "140"))
THRESH = float(os.environ.get("THRESH", "1.0"))
LAYERS = (8, 14, 20)
OUT = os.environ.get("OUT_DIR", "results_chat_demo")
EOS = "<｜end▁of▁sentence｜>"
LOG = []
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.append(s)
    os.makedirs(OUT, exist_ok=True); open(os.path.join(OUT, "live.log"), "w").write("\n".join(LOG))

# The conversation Claude authors as the human. Facts up front, casual filler, then recall.
CONVO = [
    {"u": "Hey, I'm spinning up a small project and want to keep you in the loop. First thing to remember: the project codename is Bluefin."},
    {"u": "Two more details to lock in: our launch date is October 9th, and the total budget is a hard cap of 47,000 dollars."},
    {"u": "Cool, thanks. Switching gears completely — I had an amazing bowl of tonkotsu ramen for lunch. Are you a ramen fan?"},
    {"u": "Haha fair. The weather here has been grey and rainy all week, kind of draining honestly."},
    {"u": "I've been trying to read a paper book before bed instead of doomscrolling. Slow progress though."},
    {"u": "My neighbor just got a golden retriever puppy. Adorable, but it has chaotic energy at 6am."},
    {"u": "I'm also tempted to repaint my home office a soft sage green. Supposed to be calming."},
    {"u": "Anyway — back to business. Quick memory check: what's the project codename again?", "expect": ["bluefin"]},
    {"u": "Nice. And what launch date did I give you?", "expect": ["october 9", "oct 9", "october 9th"]},
    {"u": "Last one: what's the hard budget cap I mentioned?", "expect": ["47,000", "47000", "47 000"]},
]


def emb(embT, ids):
    if ids:
        return embT(torch.tensor([ids], device=DEV))
    return torch.zeros(1, 0, embT.weight.shape[1], dtype=embT.weight.dtype, device=DEV)


@torch.no_grad()
def reply(tok, llm, pooler, embT, gate, feed, arm):
    """Generate ONE assistant reply given the full conversation `feed` (ending at
    <｜Assistant｜>). Returns (reply_ids, info)."""
    eos = tok.eos_token_id
    gen, kept, absorbed = [], [], 0
    fi = newtok = 0; done = False; n_fire = 0; pulls = []
    use_recall = arm.startswith("RECALL")
    full = (arm == "FULL")
    archive = BlockArchive(llm, tok, layers=LAYERS, block=BLOCK, mode="qk") if use_recall else None
    rec_emb = None
    gq = {}; hooks = []
    if use_recall:
        for li in LAYERS:
            def mk(li):
                def fn(_m, _i, o): gq[li] = o.detach()[0, -1].float().cpu().numpy().ravel()
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
    while not done:
        c0 = len(gen); R = c0 if full else min(c0, RW); nd = c0 - R
        if nd > absorbed:
            if archive is not None:
                archive.extend(gen[absorbed:nd])
            kept.extend(gen[absorbed:nd]); absorbed = nd
        if use_recall and (archive.blocks or len(archive.buf) >= 16):
            s = gscore()
            if s is not None and s > gate["thresh"]:
                tail = gen[-48:] if len(gen) >= 4 else feed
                rid = archive.retrieve(tail, k=RECALL_K)
                if rid:
                    rec_emb = emb(embT, rid); n_fire += 1
                    pulls.append({"at": len(gen) - len(feed), "score": round(s, 2),
                                  "text": tok.decode(rid)[:140]})
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
                if t == eos or newtok >= TURN_CAP:
                    done = True; break
                newtok += 1
            gen.append(t)
            last = llm(inputs_embeds=emb(embT, [t]), past_key_values=cache,
                       use_cache=True).logits[:, -1, :].float()
        if done:
            break
    for h in hooks:
        h.remove()
    return gen[len(feed):], {"n_fire": n_fire, "pulls": pulls}


def strip_think(text):
    return re.sub(r"<think>.*?</think>", "[…think…]", text, flags=re.S).strip()


def run_arm(tok, llm, pooler, embT, gate, arm):
    history = []
    transcript = []; correct = 0; total = 0
    for k, turn in enumerate(CONVO):
        u = turn["u"]
        feed = history + tok.encode((EOS if history else "") + f"<｜User｜>{u}<｜Assistant｜>",
                                    add_special_tokens=False)
        rid, info = reply(tok, llm, pooler, embT, gate, feed, arm)
        history = feed + rid
        rtext = tok.decode(rid)
        rec = {"turn": k, "user": u, "assistant": rtext, "assistant_clean": strip_think(rtext),
               "n_fire": info["n_fire"], "pulls": info["pulls"]}
        if "expect" in turn:
            total += 1
            norm = rtext.lower().replace(",", "").replace(" ", "")
            hit = any(e.replace(",", "").replace(" ", "") in norm for e in turn["expect"])
            rec["expect"] = turn["expect"]; rec["recall_ok"] = hit
            correct += int(hit)
        transcript.append(rec)
        log(f"  [{arm}] turn{k} fires={info['n_fire']}"
            + (f" recall_ok={rec.get('recall_ok')}" if 'expect' in turn else ""))
    return {"arm": arm, "recall_score": f"{correct}/{total}", "transcript": transcript}


def load_gate(path, thresh=None):
    z = np.load(path)
    g = {"coef": z["coef"].ravel(), "b": float(np.ravel(z["intercept"])[0]),
         "mean": z["mean"], "scale": z["scale"],
         "thresh": float(z["thresh"]) if "thresh" in z.files else 0.0}
    if thresh is not None:
        g["thresh"] = thresh
    return g


def main():
    t0 = time.time(); os.makedirs(OUT, exist_ok=True)
    tok = AutoTokenizer.from_pretrained("fft_hf")
    try:
        llm = AutoModelForCausalLM.from_pretrained("fft_hf", torch_dtype=torch.float32)
    except TypeError:
        llm = AutoModelForCausalLM.from_pretrained("fft_hf", dtype=torch.float32)
    llm = llm.eval().to(DEV)
    pooler = load_pooler().to(DEV).eval(); embT = llm.get_input_embeddings()
    gate = load_gate("gate_new.npz", thresh=THRESH)
    log(f"[demo] RW={RW} turn_cap={TURN_CAP} new_gate@thresh={THRESH} turns={len(CONVO)}")
    out = {"thresh": THRESH, "arms": {}}
    for arm in ["FULL", "SP", f"RECALL@{THRESH:g}"]:
        g = gate if arm.startswith("RECALL") else None
        log(f"--- running arm {arm} ---")
        r = run_arm(tok, llm, pooler, embT, g, arm)
        out["arms"][arm] = r
        log(f"  {arm}: recall {r['recall_score']}")
        json.dump(out, open(os.path.join(OUT, "results.json"), "w"), indent=1, ensure_ascii=False)
    out["final"] = True
    json.dump(out, open(os.path.join(OUT, "results.json"), "w"), indent=1, ensure_ascii=False)
    open(os.path.join(OUT, ".done"), "w").write("ok")
    log(f"DONE t={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
