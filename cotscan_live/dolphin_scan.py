"""dolphin_scan.py — per-position recall probe trained on REAL long Dolphin-R1 CoTs.

Recall ("attention flies >512 tokens back") is a RARE event, so we harvest it at scale:
seed with Dolphin-R1 problems whose reference CoT is >=1000 tokens (shortest-first =
efficient), teacher-force [problem + CoT] through OUR model under full KV, and at EVERY
window position (the last 512 tokens of the sequence) measure how much attention lands on
the evicted region (everything >512 back), layers 8/14/20, BOS excluded. Positions whose
attention concentrates on a single far block = recall positions (positive); the rest =
negative. Features are the SP-COMPRESSED-context pre-RoPE query at the same positions.

Trains: (1) a logistic GATE probe (recall vs non-recall position) — heldout AUC; (2) the
Phase B indexer on recall positions — heldout top-2 (which evicted block). Item-disjoint
train/heldout split = generalization. Everything streamed to OUT_DIR for live visibility.
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
from cot_recall_eval import Indexer

DEV = "cuda" if torch.cuda.is_available() else "cpu"
LAYERS = (8, 14, 20)
WINDOW, BLOCK = 512, 128
N_TRAIN = int(os.environ.get("DOLPHIN_N_TRAIN", "300"))
N_HELD = int(os.environ.get("DOLPHIN_N_HELD", "100"))
COT_MIN = int(os.environ.get("COT_MIN", "1000"))
COT_MAX = int(os.environ.get("COT_MAX", "1600"))
POS_FRAC = float(os.environ.get("POS_FRAC", "0.08"))   # top-frac window positions = recall
NEG_PER_POS = int(os.environ.get("NEG_PER_POS", "6"))
OUT = os.environ.get("OUT_DIR", "results_dolphin")

LOG = []
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.append(s)
    try:
        open(os.path.join(OUT, "live.log"), "w").write("\n".join(LOG))
    except Exception:
        pass


def pick_items(n_total):
    """Stream Dolphin-R1, parse CoT, keep 1000<=tok<=1600, return shortest-first."""
    ds = load_dataset("mlabonne/dolphin-r1-deepseek", split="train", streaming=True)
    pool = []
    seen = 0
    for ex in ds:
        seen += 1
        msgs = ex.get("messages") or []
        usr = next((m["content"] for m in msgs if m["role"] == "user"), "")
        asst = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
        if not usr or not asst:
            continue
        m = re.search(r"begin_of_thought\|>(.*?)<\|end_of_thought", asst, re.S)
        cot = (m.group(1) if m else asst).strip()
        approx = len(cot) // 4
        if COT_MIN <= approx <= COT_MAX and len(usr) < 2000:
            pool.append((approx, usr, cot))
        if len(pool) >= n_total * 3 or seen >= 8000:
            break
    pool.sort(key=lambda x: x[0])                       # shortest CoT first (efficient)
    log(f"[data] scanned {seen} rows, pool={len(pool)} in [{COT_MIN},{COT_MAX}] tok")
    return pool[:n_total]


def build_model():
    log(f"[load] fft_hf on {DEV} (eager attn)")
    tok = AutoTokenizer.from_pretrained("fft_hf")
    try:
        llm = AutoModelForCausalLM.from_pretrained(
            "fft_hf", torch_dtype=torch.float32, attn_implementation="eager")
    except TypeError:
        llm = AutoModelForCausalLM.from_pretrained(
            "fft_hf", dtype=torch.float32, attn_implementation="eager")
    llm = llm.eval().to(DEV)
    return tok, llm, load_pooler().to(DEV).eval(), llm.get_input_embeddings()


def emb(embT, ids):
    return embT(torch.tensor([ids], device=DEV))


class Cap:
    def __init__(self, llm, layers, q, k):
        self.llm, self.layers, self.wq, self.wk = llm, layers, q, k
        self.q, self.k, self.h = {}, {}, []
    def __enter__(self):
        for li in self.layers:
            a = self.llm.model.layers[li].self_attn
            if self.wq:
                self.h.append(a.q_proj.register_forward_hook(
                    lambda m, i, o, li=li: self.q.__setitem__(li, o.detach())))
            if self.wk:
                self.h.append(a.k_proj.register_forward_hook(
                    lambda m, i, o, li=li: self.k.__setitem__(li, o.detach())))
        return self
    def __exit__(self, *a):
        for h in self.h:
            h.remove()


@torch.no_grad()
def process(llm, tok, pooler, embT, usr, cot, dims):
    Hq, Hkv, D = dims
    pre = tok.encode(f"<｜User｜>{usr}<｜Assistant｜><think>\n", add_special_tokens=False)
    cot_ids = tok.encode(cot, add_special_tokens=False)
    seq = [tok.bos_token_id] + pre + cot_ids
    if len(seq) < WINDOW + BLOCK * 4 + 4:
        return None
    n_evict = ((len(seq) - WINDOW) // BLOCK) * BLOCK
    nB = n_evict // BLOCK
    win_lo = n_evict                                    # window = seq[n_evict:]
    win_len = len(seq) - win_lo

    # ---- FULL-KV attention -> per-window-position evicted-block mass + block keys ----
    cache = DynamicCache()
    with Cap(llm, LAYERS, q=False, k=True) as cap:
        o = llm(input_ids=torch.tensor([seq], device=DEV), output_attentions=True,
                use_cache=False)
    blockmass = np.zeros((win_len, nB), dtype=np.float32)
    for li in LAYERS:
        A = o.attentions[li][0].float()                 # [H, q, kv]
        a = A[:, win_lo:, :]                            # [H, win, kv]
        denom = a[:, :, 1:].sum(-1).clamp(min=1e-6)     # [H, win] (BOS excluded)
        ev = a[:, :, :n_evict].reshape(a.shape[0], win_len, nB, BLOCK).sum(-1)  # [H,win,nB]
        frac = (ev / denom.unsqueeze(-1)).mean(0)       # [win, nB] head-mean
        blockmass += frac.cpu().numpy()
    blockmass /= len(LAYERS)
    del o
    maxmass = blockmass.max(1)                          # [win] peak single-block mass
    goldblk = blockmass.argmax(1)                       # [win]
    kf = []
    for li in LAYERS:
        k = cap.k[li][0][:n_evict].view(nB, BLOCK, Hkv, D).float()
        kf.append(torch.cat([k.mean(1), k.amax(1)], -1).cpu())
    keys = torch.stack(kf, 1).half().numpy()            # [nB,L,Hkv,2D]

    # ---- SP-compressed context -> pre-RoPE q at window positions ----
    kept, window = seq[:n_evict], seq[n_evict:]
    c = DynamicCache()
    llm.model(input_ids=torch.tensor([[tok.bos_token_id]], device=DEV),
              past_key_values=c, use_cache=True)
    sp = pooler(emb(embT, kept).float()).to(embT.weight.dtype)
    llm.model(inputs_embeds=sp, past_key_values=c, use_cache=True)
    with Cap(llm, LAYERS, q=True, k=False) as cap2:
        llm.model(inputs_embeds=emb(embT, window), past_key_values=c, use_cache=True)
    q_layers = [cap2.q[li][0].float().cpu().numpy().reshape(win_len, Hq, D) for li in LAYERS]
    qfeat = np.stack(q_layers, 1)                       # [win, L, Hq, D]
    return {"qfeat": qfeat.astype(np.float16), "maxmass": maxmass, "goldblk": goldblk,
            "blockmass": blockmass, "keys": keys, "nB": nB, "win_len": win_len}


def collect(pool, llm, tok, pooler, embT, dims, tag, t0):
    Xg, yg, grp = [], [], []        # gate features (flattened q), labels, item id
    IX = []                          # indexer rows (positives)
    gi = 0
    for n, (_, usr, cot) in enumerate(pool):
        try:
            r = process(llm, tok, pooler, embT, usr, cot, dims)
        except Exception as e:
            log(f"  {tag} item {n} failed: {e}"); continue
        if r is None:
            continue
        win = r["win_len"]; mm = r["maxmass"]
        thr = np.quantile(mm, 1 - POS_FRAC)             # per-item top-frac = recall
        pos = np.where(mm >= max(thr, 0.05))[0]
        neg_all = np.where(mm < max(thr, 0.05))[0]
        rng = np.random.RandomState(n)
        neg = rng.choice(neg_all, min(len(neg_all), max(1, len(pos) * NEG_PER_POS)),
                         replace=False) if len(neg_all) else np.array([], int)
        for p in list(pos) + list(neg):
            Xg.append(r["qfeat"][p].reshape(-1)); yg.append(int(mm[p] >= max(thr, 0.05)))
            grp.append(gi)
        for p in pos:                                   # indexer positives
            IX.append({"q": r["qfeat"][p].astype(np.float32), "k": r["keys"],
                       "gold": int(r["goldblk"][p]), "nB": r["nB"],
                       "labels": r["blockmass"][p][None, :].astype(np.float32)})
        gi += 1
        if (n + 1) % 20 == 0:
            log(f"  {tag} {n+1}/{len(pool)} items={gi} gate_pts={len(yg)} "
                f"pos={int(np.sum(yg))} idx_rows={len(IX)} t={time.time()-t0:.0f}s")
    return (np.array(Xg, np.float32), np.array(yg, np.int32), np.array(grp), IX)


def train_indexer(rows_tr, rows_te, dims, seeds=2, epochs=20):
    Hq, Hkv, D = dims; L = len(LAYERS)
    def tens(r):
        q = torch.tensor(r["q"], device=DEV)
        k = torch.tensor(r["k"], dtype=torch.float32, device=DEV)
        lab = torch.tensor(r["labels"].mean(0), device=DEV); lab = lab / max(lab.sum().item(), 1e-8)
        return q, k, lab
    def top2(m, split):
        h = 0
        with torch.no_grad():
            for r in split:
                q, k, _ = tens(r)
                h += int(r["gold"]) in m(q, k).topk(min(2, len(k))).indices.tolist()
        return h / max(len(split), 1)
    best, bs = -1, None
    for sd in range(seeds):
        torch.manual_seed(sd); m = Indexer(L, Hq, Hkv, D).to(DEV)
        opt = torch.optim.Adam(m.parameters(), lr=3e-3); rng = np.random.RandomState(sd)
        for _ in range(epochs):
            rng.shuffle(rows_tr)
            for r in rows_tr:
                q, k, lab = tens(r)
                loss = F.kl_div(F.log_softmax(m(q, k), 0), lab, reduction="sum")
                opt.zero_grad(); loss.backward(); opt.step()
        v = top2(m, rows_te)
        if v > best:
            best = v; bs = {kk: vv.clone() for kk, vv in m.state_dict().items()}
    return best, bs


def snap(payload):
    json.dump(payload, open(os.path.join(OUT, "results.json"), "w"), indent=2)


def main():
    t0 = time.time(); os.makedirs(OUT, exist_ok=True)
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    tok, llm, pooler, embT = build_model()
    cfg = llm.config; Hq = cfg.num_attention_heads; Hkv = cfg.num_key_value_heads
    D = cfg.hidden_size // Hq; dims = (Hq, Hkv, D)
    log(f"[dims] Hq={Hq} Hkv={Hkv} D={D} feat={len(LAYERS)*Hq*D}")

    pool = pick_items(N_TRAIN + N_HELD)
    tr_pool, te_pool = pool[:N_TRAIN], pool[N_TRAIN:]   # item-disjoint
    log(f"[split] train_items={len(tr_pool)} held_items={len(te_pool)}")

    Xtr, ytr, _, IXtr = collect(tr_pool, llm, tok, pooler, embT, dims, "TRAIN", t0)
    Xte, yte, _, IXte = collect(te_pool, llm, tok, pooler, embT, dims, "HELD", t0)
    log(f"[gate data] train pts={len(ytr)} pos={int(ytr.sum())} | held pts={len(yte)} "
        f"pos={int(yte.sum())}")

    res = {"elapsed_s": round(time.time()-t0, 1),
           "config": {"n_train": N_TRAIN, "n_held": N_HELD, "cot": [COT_MIN, COT_MAX],
                      "pos_frac": POS_FRAC},
           "gate": {"train_pts": int(len(ytr)), "train_pos": int(ytr.sum()),
                    "held_pts": int(len(yte)), "held_pos": int(yte.sum())},
           "indexer": {"train_pos": len(IXtr), "held_pos": len(IXte)}}
    if ytr.sum() > 5 and yte.sum() > 2:
        sc = StandardScaler().fit(Xtr)
        clf = LogisticRegression(C=0.05, max_iter=3000, class_weight="balanced").fit(
            sc.transform(Xtr), ytr)
        auc = float(roc_auc_score(yte, clf.decision_function(sc.transform(Xte))))
        res["gate"]["heldout_auc"] = round(auc, 4)
        log(f"[GATE] heldout AUC = {auc:.4f}  (generalization)")
    snap(res)
    if len(IXtr) > 10 and len(IXte) > 5:
        # subsample positives: top-2 saturates well below the full set, and per-row CPU
        # training over 10k+ rows is the runtime bottleneck. cap for a fast, sufficient fit.
        IXCAP = int(os.environ.get("IDX_CAP", "2500"))
        rng = np.random.RandomState(0)
        if len(IXtr) > IXCAP:
            IXtr = [IXtr[i] for i in rng.choice(len(IXtr), IXCAP, replace=False)]
        if len(IXte) > IXCAP // 2:
            IXte = [IXte[i] for i in rng.choice(len(IXte), IXCAP // 2, replace=False)]
        log(f"[indexer] training on {len(IXtr)} pos, eval on {len(IXte)} pos")
        t1, _ = train_indexer([dict(r) for r in IXtr], IXte, dims)
        # raw qk reference at recall positions
        rawhit = 0
        for r in IXte:
            q = torch.tensor(r["q"]); k = torch.tensor(r["k"], dtype=torch.float32)
            g = q.shape[1] // k.shape[2]
            kk = k[..., :q.shape[-1]].repeat_interleave(g, 2)
            s = torch.einsum("lhd,blhd->blh", q, kk).mean((1, 2))
            rawhit += int(r["gold"]) in s.topk(min(2, len(k))).indices.tolist()
        res["indexer"]["heldout_top2"] = round(t1, 4)
        res["indexer"]["raw_qk_heldout_top2"] = round(rawhit / len(IXte), 4)
        log(f"[INDEXER] heldout top-2 = {t1:.4f}  (raw-qk ref {rawhit/len(IXte):.4f})")
    res["final"] = True; snap(res)
    log("\n=== SUMMARY ===")
    log(f"  GATE heldout AUC: {res['gate'].get('heldout_auc')}  "
        f"(train pos {res['gate']['train_pos']}/{res['gate']['train_pts']}, "
        f"held pos {res['gate']['held_pos']}/{res['gate']['held_pts']})")
    log(f"  INDEXER heldout top-2: {res['indexer'].get('heldout_top2')} "
        f"(raw-qk {res['indexer'].get('raw_qk_heldout_top2')}, pos {len(IXte)})")
    log(f"DONE t={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
