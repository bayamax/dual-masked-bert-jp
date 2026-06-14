"""chat_solve.py — GENERATION-time recall eval on conversational multi-turn (雑談).

Needle-in-a-conversation: a specific fact is stated early, buried under casual turns until it
is evicted past the exposure window (so it lives only in the SP gist), then the final user turn
asks for it. We greedily generate the assistant answer under four arms on the SAME fft_hf model:

  FULL          full KV, no compression          (needle still in context = upper bound)
  SP            SP gist + window, NO recall       (needle gone = should fail)
  RECALL_new    SP + gate(NEW, mixed-trained) + raw-QK retrieval of the evicted blocks
  RECALL_old    SP + gate(OLD, dolphin-only)  + raw-QK retrieval (same retriever)

The two RECALL arms share the identical retriever and differ ONLY in the gate, so the
comparison isolates the retrained gate's effect on generation-time recall. Metrics per arm:
  acc        : the needle value appears in the generated answer
  fire_rate  : the gate fired at least once during generation
  pull_rate  : a retrieved block actually contained the needle value
"""
import os, re, json, time, random
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
C, CAP, BLOCK, RECALL_K = 64, int(os.environ.get("CAP", "200")), 128, 2
LAYERS = (8, 14, 20)
N = int(os.environ.get("CHAT_N", "24"))
OUT = os.environ.get("OUT_DIR", "results_chat")
LOG = []
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.append(s)
    os.makedirs(OUT, exist_ok=True); open(os.path.join(OUT, "live.log"), "w").write("\n".join(LOG))

NUM_RE = re.compile(r"\d{2,6}")

NEEDLES = [
    ("the office wifi password is {v}", "wifi password", 1000, 9999),
    ("my locker number is {v}", "locker number", 100, 999),
    ("the meeting is in room {v}", "room number", 100, 999),
    ("my membership id is {v}", "membership id", 10000, 99999),
    ("the parking spot is number {v}", "parking spot number", 10, 99),
    ("the safe combination is {v}", "safe combination", 1000, 9999),
    ("my reference code is {v}", "reference code", 10000, 99999),
    ("the front gate code is {v}", "gate code", 1000, 9999),
]
FILLER = [
    ("How's your day going so far?", "Pretty good, thanks for asking! Just keeping busy. How about you?"),
    ("I'm thinking of trying a new recipe tonight.", "Ooh, nice! What are you planning to make?"),
    ("The weather has been so unpredictable lately.", "Right? One minute sun, the next it's pouring."),
    ("I watched a great documentary yesterday.", "Love a good documentary. What was it about?"),
    ("My cat keeps knocking things off the table.", "Haha, classic cat behavior. They love the drama."),
    ("I really need to start exercising more.", "Same here. Even a short daily walk helps a lot."),
    ("Do you have any book recommendations?", "Depends on your mood — fiction or something practical?"),
    ("I'm trying to drink more water these days.", "Good goal! Keeping a bottle nearby makes it easier."),
    ("Traffic this morning was a nightmare.", "Ugh, the worst way to start a day. Hope it's clear now."),
    ("I might repaint my living room soon.", "Fun project! Any color in mind?"),
    ("Coffee or tea — what's your pick?", "Coffee in the morning, tea in the evening, honestly."),
    ("I keep forgetting to water my plants.", "A little phone reminder can save them, trust me."),
    ("Weekend plans are looking pretty relaxed.", "Sometimes a quiet weekend is exactly what you need."),
    ("I started learning to play guitar.", "That's awesome! Are you using lessons or just winging it?"),
    ("The new cafe downtown is really cozy.", "I love discovering a good cozy spot. Good pastries?"),
    ("I've been listening to a lot of jazz lately.", "Jazz is perfect background music for focusing."),
    ("My phone battery dies way too fast now.", "Might be time for a battery swap — they wear out."),
    ("I want to take a short trip next month.", "Nice! Somewhere new or an old favorite?"),
    ("I finally cleaned out my closet.", "That always feels so satisfying afterward."),
    ("The sunset yesterday was unbelievable.", "Those skies make you stop and just stare, don't they?"),
    ("I'm not great at remembering names.", "You're not alone — repeating it once out loud helps."),
    ("I think I'll bake some bread this weekend.", "Fresh bread smell beats almost everything. Enjoy!"),
    ("My neighbor got a new puppy.", "Puppy energy is unmatched. Cute but exhausting!"),
    ("I've been meaning to organize my photos.", "A rainy afternoon is perfect for that kind of task."),
]
EOS = "<｜end▁of▁sentence｜>"


def build(seed):
    r = random.Random(seed)
    tmpl, kw, lo, hi = NEEDLES[seed % len(NEEDLES)]
    val = str(r.randint(lo, hi))
    needle_u = "Oh, before I forget — " + tmpl.format(v=val) + "."
    needle_a = "Got it, I'll keep that in mind."
    fills = r.sample(FILLER, len(FILLER))
    turns = [(needle_u, needle_a)]
    i = 0
    while sum(len((u + a)) for u, a in turns) // 4 < RW + 200:
        turns.append(fills[i % len(fills)]); i += 1
    final_u = f"Quick question — what was the {kw} I mentioned earlier? Just give me the number."
    return turns, final_u, val, kw


def to_feed(tok, turns, final_u):
    ids = []
    for u, a in turns:
        ids += tok.encode((EOS if ids else "") + f"<｜User｜>{u}<｜Assistant｜>{a}",
                          add_special_tokens=False)
    ids += tok.encode(EOS + f"<｜User｜>{final_u}<｜Assistant｜>", add_special_tokens=False)
    return ids


def emb(embT, ids):
    if ids:
        return embT(torch.tensor([ids], device=DEV))
    return torch.zeros(1, 0, embT.weight.shape[1], dtype=embT.weight.dtype, device=DEV)


@torch.no_grad()
def solve(tok, llm, pooler, embT, gate, feed, value, arm):
    eos = tok.eos_token_id
    gen, kept, absorbed = list(), [], 0
    fi = new = 0; done = False; max_kept = 0; n_fire = 0; pull_hit = False
    use_recall = arm.startswith("RECALL")
    full = (arm == "FULL")
    archive = BlockArchive(llm, tok, layers=LAYERS, block=BLOCK, mode="qk") if use_recall else None
    rec_emb = None; trace = []
    gq = {}; hooks = []
    if use_recall:
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
    while not done:
        c0 = len(gen); R = c0 if full else min(c0, RW); nd = c0 - R
        if nd > absorbed:
            if archive is not None:
                archive.extend(gen[absorbed:nd])
            kept.extend(gen[absorbed:nd]); absorbed = nd; max_kept = max(max_kept, len(kept))
        if use_recall and (archive.blocks or len(archive.buf) >= 16):
            s = gscore()
            if s is not None and s > gate["thresh"]:
                tail = gen[-48:] if len(gen) >= 4 else feed
                rid = archive.retrieve(tail, k=RECALL_K)
                if rid:
                    rec_emb = emb(embT, rid); n_fire += 1
                    pulled = tok.decode(rid)
                    if value in pulled.replace(",", ""):
                        pull_hit = True
                    trace.append({"at_tok": len(gen) - len(feed), "score": round(s, 2),
                                  "pull_hit": value in pulled.replace(",", ""),
                                  "cot_here": tok.decode(gen[-24:])})
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
    answer = tok.decode(gen[len(feed):])
    correct = value in answer.replace(",", "")
    return {"answer": answer, "correct": correct, "n_fire": n_fire, "pull_hit": pull_hit,
            "kept": max_kept, "tokens": len(gen) - len(feed), "trace": trace}


def load_gate(path):
    z = np.load(path)
    return {"coef": z["coef"].ravel(), "b": float(np.ravel(z["intercept"])[0]),
            "mean": z["mean"], "scale": z["scale"],
            "thresh": float(z["thresh"]) if "thresh" in z.files else 0.0}


def main():
    t0 = time.time(); os.makedirs(OUT, exist_ok=True)
    tok = AutoTokenizer.from_pretrained("fft_hf")
    try:
        llm = AutoModelForCausalLM.from_pretrained("fft_hf", torch_dtype=torch.float32)
    except TypeError:
        llm = AutoModelForCausalLM.from_pretrained("fft_hf", dtype=torch.float32)
    llm = llm.eval().to(DEV)
    pooler = load_pooler().to(DEV).eval(); embT = llm.get_input_embeddings()
    gate_new = load_gate("gate_new.npz") if os.path.exists("gate_new.npz") else None
    gate_old = load_gate("gate_old.npz") if os.path.exists("gate_old.npz") else None
    sweep = [float(x) for x in os.environ.get("THRESH_SWEEP", "2.0,1.0,0.0").split(",")]
    log(f"[chat] N={N} RW={RW} cap={CAP} | new_thresh(default)={gate_new and round(gate_new['thresh'],2)} "
        f"old_thresh={gate_old and round(gate_old['thresh'],2)} | sweep(new)={sweep}")
    arms = [("FULL", None), ("SP", None)]
    if gate_old is not None:
        arms.append(("RECALL_old", gate_old))               # old gate at its own trained thresh
    for t in sweep:                                          # new gate, swept threshold
        g = dict(gate_new); g["thresh"] = t
        arms.append((f"RECALL_new@{t:g}", g))
    agg = {a: {"correct": 0, "fire": 0, "pull": 0} for a, _ in arms}
    results = []
    for i in range(N):
        turns, final_u, val, kw = build(i)
        feed = to_feed(tok, turns, final_u)
        rec = {"i": i, "value": val, "kw": kw, "feed_tokens": len(feed), "turns": len(turns), "arms": {}}
        for a, g in arms:
            m = solve(tok, llm, pooler, embT, g, feed, val, a)
            agg[a]["correct"] += int(m["correct"]); agg[a]["fire"] += int(m["n_fire"] > 0)
            agg[a]["pull"] += int(m["pull_hit"])
            rec["arms"][a] = {"correct": m["correct"], "n_fire": m["n_fire"], "pull_hit": m["pull_hit"],
                              "tokens": m["tokens"], "answer": m["answer"][:240], "trace": m["trace"]}
        results.append(rec)
        line = " | ".join(
            f"{a}={'OK' if rec['arms'][a]['correct'] else 'x'}"
            + (f"(f{rec['arms'][a]['n_fire']},{'P' if rec['arms'][a]['pull_hit'] else '-'})" if a.startswith('RECALL') else '')
            for a, _ in arms)
        log(f"[{i:2}] {kw}={val} feedtok={len(feed)} :: {line}  [{i+1}/{N} t={time.time()-t0:.0f}s]")
        n = i + 1
        summ = {a: {"acc": f"{agg[a]['correct']}/{n}", "fire": f"{agg[a]['fire']}/{n}",
                    "pull": f"{agg[a]['pull']}/{n}"} for a, _ in arms}
        json.dump({"n": n, "summary": summ, "results": results}, open(os.path.join(OUT, "results.json"), "w"), indent=1)
    n = N
    summ = {a: {"acc": f"{agg[a]['correct']}/{n} = {agg[a]['correct']/n:.0%}",
                "fire_rate": f"{agg[a]['fire']}/{n} = {agg[a]['fire']/n:.0%}",
                "pull_rate": f"{agg[a]['pull']}/{n} = {agg[a]['pull']/n:.0%}"} for a, _ in arms}
    log("\n=== CHAT generation recall (needle-in-conversation) ===")
    for a, _ in arms:
        log(f"  {a:11}: acc {summ[a]['acc']:>14} | fire {summ[a].get('fire_rate','-'):>14} | pull {summ[a].get('pull_rate','-'):>14}")
    json.dump({"n": n, "summary": summ, "results": results, "final": True},
              open(os.path.join(OUT, "results.json"), "w"), indent=1)
    open(os.path.join(OUT, ".done"), "w").write("ok")
    log(f"DONE t={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
