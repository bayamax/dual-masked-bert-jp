"""recall_gen.py — end-to-end MULTI-FACT recall eval in chat generation.

The single-fact demo exposed a retrieval failure: with several facts in DIFFERENT evicted
blocks, raw-QK over-pulls the opening block. This benchmark plants 3 distinct facts in 3 early
turns, buries them past the window, then asks each one back. Arms (same contrastive gate; differ
only in retriever) on the SAME model:

  SP        SP gist + window, no recall (lower bound)
  RC_raw    gate + raw-QK BlockArchive retrieval (current)
  RC_idx    gate + LEARNED QK indexer retrieval (top-2 0.998 on held)
  RC_bge    gate + BGE-on-demand retrieval (on-device target; IDs-only)

Arm label grammar:  BASE[@THRESH][:rFLOOR]
  @THRESH  gate firing threshold (e.g. RC_bge@-3.5)
  :rFLOOR  relevance gate — only INJECT if the top retrieval score clears FLOOR (e.g. RC_bge@-3.5:r0.2)

Env: Q4=1 loads the base model in 4-bit (nf4) for on-device evaluation; GATE_NPZ selects the gate
file. Per arm we also dump the per-fire top retrieval scores split into q-fires (recall turns) vs
c-fires (casual probe turns) — the calibration data for the relevance floor.

Fact/filler turns are teacher-forced (canned replies) to grow context cheaply; only question and
casual-probe turns are generated. Metrics per arm: recall acc (value in answer), retrieval hit
(value in the pulled block), and casual false-fire rate (fires on unrelated probe turns).
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
try:
    from cot_recall_eval import Indexer
except Exception:
    Indexer = None

DEV = "cuda" if torch.cuda.is_available() else "cpu"
RW = int(os.environ.get("RW", "512"))
C, BLOCK, RECALL_K = 64, 128, int(os.environ.get("RECALL_K", "2"))
TURN_CAP = int(os.environ.get("TURN_CAP", "110"))
THRESH = float(os.environ.get("THRESH", "-2.858"))
N = int(os.environ.get("GEN_N", "4"))
ARMS = os.environ.get("ARMS", "SP,RC_raw,RC_idx").split(",")
LAYERS = (8, 14, 20)
OUT = os.environ.get("OUT_DIR", "results_recall_gen")
GATE_NPZ = os.environ.get("GATE_NPZ", "gate_new.npz")
Q4 = os.environ.get("Q4", "0") == "1"
EOS = "<｜end▁of▁sentence｜>"
LOG = []
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.append(s)
    os.makedirs(OUT, exist_ok=True); open(os.path.join(OUT, "live.log"), "w").write("\n".join(LOG))

FACTS = [
    ("project codename", "Bluefin"), ("vault code", "7382"), ("flight number", "JL412"),
    ("hotel confirmation number", "88231"), ("locker number", "514"), ("meeting room", "C12"),
    ("reference code", "ZX9043"), ("account pin", "6207"), ("server name", "atlas03"),
    ("badge id", "40718"), ("table number", "23"), ("invoice number", "55920"),
]
CASUAL = [
    ("How's your day going so far?", "Pretty good, thanks for asking! It's been a steady kind of day, nothing too dramatic. I got through my morning tasks earlier than expected, so I'm feeling fairly relaxed about the afternoon. How has yours been treating you?"),
    ("I had great ramen for lunch today.", "Oh that sounds wonderful, ramen is such a comforting meal. Tonkotsu broth on a cool day is honestly hard to beat, especially with a soft egg and some extra scallions. Did you go to a regular spot or try somewhere new this time?"),
    ("The weather's been grey and rainy all week.", "Ugh, I hear you, long stretches of grey skies can really weigh on your mood after a while. It makes it so tempting to just stay curled up indoors with something warm. Hopefully there's a bit of sunshine coming to break it up soon."),
    ("I'm trying to read more before bed lately.", "That's a really nice habit to build, much kinder to your brain than scrolling on a bright screen late at night. Even a few pages can help you wind down and sleep better. What kind of books are you reaching for these days?"),
    ("My neighbor just got a golden retriever puppy.", "Aw, that's adorable, golden retriever puppies are pure sunshine, though they do come with a serious amount of chaotic energy. Lots of zoomies and chewed shoes in their future, I'd guess. I bet your neighborhood walks just got a lot more entertaining."),
    ("I'm thinking of repainting my home office.", "Ooh, a fresh coat of paint can completely change how a room feels, that sounds like a fun little project. A calm, soft color can really help you focus and feel settled while you work. Do you already have a shade in mind or are you still browsing swatches?"),
    ("I started learning to play the guitar.", "That's awesome, picking up an instrument is such a rewarding thing to do. The first few weeks can be tough on the fingertips, but the chords start coming together faster than you'd expect once muscle memory kicks in. Are you following lessons or just figuring out songs you like?"),
    ("Traffic this morning was an absolute nightmare.", "Oh no, there's really no worse way to start the day than being stuck bumper to bumper when you just want to get where you're going. It drains your energy before anything has even happened. I hope the roads cleared up and the rest of your commute was smoother."),
    ("I tried a new coffee place downtown.", "Nice, discovering a cozy new cafe is one of life's small but genuine pleasures. There's something really satisfying about finding a spot with good coffee and a comfortable corner to sit in. Did the pastries live up to the coffee, or was it more of a drinks-only kind of place?"),
    ("I've been listening to a lot of jazz lately.", "Jazz is such a great choice, it has this wonderful way of filling a room without demanding all your attention. It makes for perfect background music whether you're working, cooking, or just relaxing in the evening. Any particular artists or albums you've had on repeat?"),
    ("I might take a short trip next month.", "That sounds refreshing, a little change of scenery can do wonders for resetting your head. Even a couple of days away can feel like a proper reset. Are you thinking of revisiting an old favorite place or exploring somewhere completely new this time?"),
    ("The sunset yesterday evening was unreal.", "I love when the sky puts on a show like that, those deep oranges and pinks that make you stop whatever you're doing just to stare. They're such a nice reminder to slow down for a moment. Did you manage to catch a photo, or was it one of those you just had to take in?"),
]


def build_problem(seed):
    r = random.Random(seed)
    fi = r.sample(range(len(FACTS)), 3)
    facts = [FACTS[i] for i in fi]
    steps = []
    for kw, val in facts:                                  # 3 distinct facts, distinct turns
        steps.append({"kind": "fact", "u": f"Important to note: the {kw} is {val}.",
                      "a": "Got it, noted."})
    fills = r.sample(CASUAL, len(CASUAL))
    for j in range(9):                                      # long filler so the facts are evicted
        u, a = fills[j % len(fills)]; steps.append({"kind": "filler", "u": u, "a": a})
    steps.append({"kind": "probe", "u": fills[9 % len(fills)][0]})   # casual probe (must NOT fire)
    steps.append({"kind": "probe", "u": fills[10 % len(fills)][0]})
    for kw, val in [facts[k] for k in r.sample(range(3), 3)]:
        steps.append({"kind": "q", "u": f"Quick — what was the {kw} I mentioned? Just the value.",
                      "kw": kw, "val": val})
    return steps


def emb(embT, ids):
    if ids:
        return embT(torch.tensor([ids], device=DEV))
    return torch.zeros(1, 0, embT.weight.shape[1], dtype=embT.weight.dtype, device=DEV)


import torch.nn as nn


class BGEHead(nn.Module):
    def __init__(self, qdim, kdim, d=128):
        super().__init__()
        self.Wq = nn.Sequential(nn.Linear(qdim, d), nn.GELU(), nn.Linear(d, d))
        self.Wk = nn.Sequential(nn.Linear(kdim, d), nn.GELU(), nn.Linear(d, d))
    def forward(self, q, k):
        return self.Wq(q) @ self.Wk(k).T


BGE_MODEL = None
BGE_HEAD = None


def retrieve_bge(tok, archive, gq, k):
    """ON-DEMAND BGE retrieval: query = the gate's hidden q (free), keys = BGE-encoded block
    TEXT computed on the fly from stored IDs only (no persisted vectors). bge_head bridges them.
    Returns (injected_ids, top_score) — top_score is the relevance signal for the floor."""
    cands = list(archive.blocks)
    if len(archive.buf) >= 16:
        cands = cands + [{"ids": list(archive.buf)}]
    if not cands or any(li not in gq for li in LAYERS):
        return [], None
    texts = [tok.decode(b["ids"]) for b in cands]
    K = BGE_MODEL.encode(texts, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
    q = np.concatenate([gq[li] for li in LAYERS]).astype(np.float32)
    with torch.no_grad():
        s = BGE_HEAD(torch.tensor(q, device=DEV), torch.tensor(K, device=DEV))
    top = s.topk(min(k, len(cands))).indices.tolist()
    out = []
    for i in sorted(top):
        out.extend(cands[i]["ids"])
    return out, float(s.max().item())


def retrieve_idx(archive, gq, indexer, dims, k):
    Hq, Hkv, D = dims
    cands = list(archive.blocks)
    if len(archive.buf) >= 16:
        kc = archive._capture(list(archive.buf), want_q=False)
        cands = cands + [{"ids": list(archive.buf),
                          "k": torch.stack([kc[li] for li in LAYERS])}]
    if not cands or any(li not in gq for li in LAYERS):
        return [], None
    Ks = []
    for b in cands:
        kk = b["k"].float()                                # [L,T,Hkv,D]
        Ks.append(torch.cat([kk.mean(1), kk.amax(1)], -1))  # [L,Hkv,2D]
    K = torch.stack(Ks, 0).to(DEV)
    q = torch.stack([torch.tensor(gq[li], dtype=torch.float32).view(Hq, D) for li in LAYERS]).to(DEV)
    with torch.no_grad():
        s = indexer(q, K)
    top = s.topk(min(k, len(cands))).indices.tolist()
    out = []
    for i in sorted(top):
        out.extend(cands[i]["ids"])
    return out, float(s.max().item())


@torch.no_grad()
def reply(tok, llm, pooler, embT, gate, indexer, dims, feed, arm, thresh, rfloor=None):
    Hq, Hkv, D = dims
    eos = tok.eos_token_id
    gen, kept, absorbed = [], [], 0
    fi = newtok = 0; done = False; n_fire = 0; pulls = []; retr_scores = []
    use_recall = arm.startswith("RC")
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
            if s is not None and s > thresh:
                if arm == "RC_idx":
                    rid, rsc = retrieve_idx(archive, gq, indexer, dims, RECALL_K)
                elif arm == "RC_bge":
                    rid, rsc = retrieve_bge(tok, archive, gq, RECALL_K)
                else:
                    rid = archive.retrieve(gen[-48:] if len(gen) >= 4 else feed, k=RECALL_K); rsc = None
                if rid:
                    retr_scores.append(rsc)                          # relevance-floor calibration
                    if (rfloor is None) or (rsc is None) or (rsc >= rfloor):
                        rec_emb = emb(embT, rid); n_fire += 1
                        pulls.append(tok.decode(rid))
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
    return gen[len(feed):], {"n_fire": n_fire, "pulls": pulls, "retr_scores": retr_scores}


def parse_arm(label):
    """BASE[@THRESH][:rFLOOR] -> (base, thresh, rfloor)."""
    spec = label; rfloor = None
    if ":r" in spec:
        spec, rf = spec.split(":r", 1); rfloor = float(rf)
    base = spec.split("@")[0]
    thresh = float(spec.split("@")[1]) if "@" in spec else float(os.environ.get("THRESH", "-2.858"))
    return base, thresh, rfloor


def run_arm(tok, llm, pooler, embT, gate, indexer, dims, steps, label):
    base, thresh, rfloor = parse_arm(label)
    history = []
    qres = []; casual_fires = 0; casual_turns = 0; first_feed = 0
    q_fires = []; c_fires = []
    for st in steps:
        if st["kind"] in ("fact", "filler"):
            history += tok.encode((EOS if history else "") + f"<｜User｜>{st['u']}<｜Assistant｜>{st['a']}",
                                  add_special_tokens=False)
            continue
        feed = history + tok.encode((EOS if history else "") + f"<｜User｜>{st['u']}<｜Assistant｜>",
                                    add_special_tokens=False)
        if not first_feed:
            first_feed = len(feed)
        rid, info = reply(tok, llm, pooler, embT, gate, indexer, dims, feed, base, thresh, rfloor)
        history = feed + rid
        rtext = tok.decode(rid)
        rs = [x for x in info["retr_scores"] if x is not None]
        if st["kind"] == "probe":
            casual_turns += 1; casual_fires += int(info["n_fire"] > 0); c_fires += rs
        else:  # question
            q_fires += rs
            norm = rtext.replace(",", "").replace(" ", "").lower()
            ok = st["val"].replace(",", "").replace(" ", "").lower() in norm
            pulled_ok = any(st["val"].replace(",", "").lower() in p.replace(",", "").lower() for p in info["pulls"])
            qres.append({"kw": st["kw"], "val": st["val"], "recall_ok": ok, "retrieved_ok": pulled_ok,
                         "n_fire": info["n_fire"], "answer": rtext[:160]})
    return qres, casual_fires, casual_turns, first_feed, q_fires, c_fires


def _pct(xs):
    if not xs:
        return None
    a = np.array(xs, np.float32)
    return {"n": len(xs), "min": round(float(a.min()), 4), "p10": round(float(np.quantile(a, 0.1)), 4),
            "median": round(float(np.median(a)), 4), "p90": round(float(np.quantile(a, 0.9)), 4),
            "max": round(float(a.max()), 4)}


def main():
    t0 = time.time(); os.makedirs(OUT, exist_ok=True)
    tok = AutoTokenizer.from_pretrained("fft_hf")
    if Q4:
        from transformers import BitsAndBytesConfig
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
        llm = AutoModelForCausalLM.from_pretrained("fft_hf", quantization_config=bnb,
                                                   device_map={"": 0}).eval()
        log("[model] loaded 4-bit (nf4) base model")
    else:
        try:
            llm = AutoModelForCausalLM.from_pretrained("fft_hf", torch_dtype=torch.float32)
        except TypeError:
            llm = AutoModelForCausalLM.from_pretrained("fft_hf", dtype=torch.float32)
        llm = llm.eval().to(DEV)
        log("[model] loaded fp32 base model")
    pooler = load_pooler().to(DEV).eval(); embT = llm.get_input_embeddings()
    cfg = llm.config; Hq = cfg.num_attention_heads; Hkv = cfg.num_key_value_heads
    D = cfg.hidden_size // Hq; dims = (Hq, Hkv, D)
    z = np.load(GATE_NPZ)
    gate = {"coef": z["coef"].ravel(), "b": float(np.ravel(z["intercept"])[0]),
            "mean": z["mean"], "scale": z["scale"], "thresh": THRESH}
    log(f"[gate] loaded {GATE_NPZ}")
    need_idx = any(parse_arm(a)[0] == "RC_idx" for a in ARMS)
    indexer = None
    if need_idx:
        if Indexer is None or not os.path.exists("indexer.npz"):
            raise RuntimeError("RC_idx requested but Indexer/indexer.npz unavailable")
        zi = np.load("indexer.npz")
        indexer = Indexer(len(LAYERS), Hq, Hkv, D).to(DEV)
        indexer.load_state_dict({"Aq": torch.tensor(zi["Aq"]), "Bk": torch.tensor(zi["Bk"]),
                                 "w": torch.tensor(zi["w"])})
        indexer.eval()
    global BGE_MODEL, BGE_HEAD
    if any(parse_arm(a)[0] == "RC_bge" for a in ARMS):
        from sentence_transformers import SentenceTransformer
        BGE_MODEL = SentenceTransformer("BAAI/bge-base-en-v1.5", device=DEV)
        zb = np.load("bge_head.npz")
        BGE_HEAD = BGEHead(int(zb["qdim"]), int(zb["kdim"]), int(zb["d"])).to(DEV)
        BGE_HEAD.load_state_dict({k: torch.tensor(zb[k]) for k in zb.files if k not in ("qdim","kdim","d")})
        BGE_HEAD.eval()
        log("[bge] loaded BAAI/bge-base-en-v1.5 + bge_head")
    log(f"[gen] N={N} RW={RW} Q4={Q4} arms={ARMS}")
    agg = {a: {"qc": 0, "qn": 0, "rc": 0, "cf": 0, "ct": 0, "qfire": [], "cfire": []} for a in ARMS}
    results = []
    for i in range(N):
        steps = build_problem(i)
        rec = {"i": i, "facts": [(s["kw"], s["val"]) for s in steps if s["kind"] == "q"], "arms": {}}
        for a in ARMS:
            qres, cf, ct, ff_len, qf, cfr = run_arm(tok, llm, pooler, embT, gate, indexer, dims, steps, a)
            qc = sum(q["recall_ok"] for q in qres); rc = sum(q["retrieved_ok"] for q in qres)
            agg[a]["qc"] += qc; agg[a]["qn"] += len(qres); agg[a]["rc"] += rc
            agg[a]["cf"] += cf; agg[a]["ct"] += ct
            agg[a]["qfire"] += qf; agg[a]["cfire"] += cfr
            rec["arms"][a] = {"recall": f"{qc}/{len(qres)}", "retrieved": f"{rc}/{len(qres)}",
                              "casual_fire": f"{cf}/{ct}", "q": qres}
        results.append(rec)
        line = " | ".join(f"{a}: rec {rec['arms'][a]['recall']} ret {rec['arms'][a]['retrieved']} ff {rec['arms'][a]['casual_fire']}" for a in ARMS)
        log(f"[{i}] feed1={ff_len}tok | {line}  [t={time.time()-t0:.0f}s]")
        json.dump({"n": i+1, "results": results}, open(os.path.join(OUT, "results.json"), "w"), indent=1)
    summ = {a: {"recall": f"{agg[a]['qc']}/{agg[a]['qn']} = {agg[a]['qc']/max(agg[a]['qn'],1):.0%}",
                "retrieved": f"{agg[a]['rc']}/{agg[a]['qn']} = {agg[a]['rc']/max(agg[a]['qn'],1):.0%}",
                "casual_fire": f"{agg[a]['cf']}/{agg[a]['ct']} = {agg[a]['cf']/max(agg[a]['ct'],1):.0%}",
                "tscore_q_fires": _pct(agg[a]["qfire"]), "tscore_c_fires": _pct(agg[a]["cfire"])} for a in ARMS}
    log("\n=== MULTI-FACT recall (chat generation) ===")
    for a in ARMS:
        log(f"  {a:16}: recall {summ[a]['recall']:>14} | retrieved {summ[a]['retrieved']:>14} | casual-fire {summ[a]['casual_fire']:>12}")
        log(f"      tscore q-fires={summ[a]['tscore_q_fires']}  c-fires={summ[a]['tscore_c_fires']}")
    json.dump({"n": N, "summary": summ, "results": results, "final": True},
              open(os.path.join(OUT, "results.json"), "w"), indent=1)
    open(os.path.join(OUT, ".done"), "w").write("ok")
    log(f"DONE t={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
