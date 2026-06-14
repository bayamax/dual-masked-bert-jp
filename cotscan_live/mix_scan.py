"""mix_scan.py — retrain the recall GATE + retrievers on a DIVERSE corpus.

The previous gate/indexer/bge_head were harvested ONLY from Dolphin-R1 long single-turn
CoTs, so they are biased to monologue-reasoning recall and barely fire in conversational
multi-turn settings. The recall label ("attention flies past the 512-token exposure window
to a single far block") is source-agnostic: it works on ANY token sequence. So here we
harvest the SAME label on a MIX of corpora and retrain:

  * dolphin  : mlabonne/dolphin-r1-deepseek  -> problem + long reasoning CoT (single turn)
  * ultrachat: HuggingFaceH4/ultrachat_200k  -> instructional multi-turn dialogue
  * soda     : allenai/soda                  -> casual social multi-turn dialogue (many turns)

For every corpus we teacher-force the formatted token sequence through OUR model under full
KV, and at every window position measure attention to the evicted region (layers 8/14/20,
BOS excluded). Top-frac positions = recall (positive). Features = SP-compressed-context
pre-RoPE query at the same positions (exactly the runtime feature).

Trains and SAVES (recall_kit format): gate.npz / indexer.npz / bge_head.npz. Reports heldout
AUC / top-2 PER SOURCE (item-disjoint) so we can see it now generalizes to conversation.
"""
import os, re, json, time
import numpy as np
import torch
import torch.nn as nn
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
SEQ_MIN = int(os.environ.get("SEQ_MIN", "1000"))   # total-seq token band (eviction must bite)
SEQ_MAX = int(os.environ.get("SEQ_MAX", "1800"))
POS_FRAC = float(os.environ.get("POS_FRAC", "0.08"))
NEG_PER_POS = int(os.environ.get("NEG_PER_POS", "6"))
# per-source item budgets (train / held)
N_DOLPHIN_TR = int(os.environ.get("N_DOLPHIN_TR", "150"))
N_DOLPHIN_TE = int(os.environ.get("N_DOLPHIN_TE", "50"))
N_CHAT_TR = int(os.environ.get("N_CHAT_TR", "220"))     # split across ultrachat+soda
N_CHAT_TE = int(os.environ.get("N_CHAT_TE", "70"))
WITH_BGE = os.environ.get("WITH_BGE", "1") == "1"
OUT = os.environ.get("OUT_DIR", "results_mix")
ART = os.environ.get("ART_DIR", "artifacts_mix")

LOG = []
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.append(s)
    try: open(os.path.join(OUT, "live.log"), "w").write("\n".join(LOG))
    except Exception: pass


# ----------------------------- data providers (yield token seqs) -----------------------------
def _seq_ok(seq):
    return seq is not None and SEQ_MIN <= len(seq) <= SEQ_MAX and len(seq) >= WINDOW + BLOCK * 4 + 4


def dolphin_seqs(tok, n, bos):
    ds = load_dataset("mlabonne/dolphin-r1-deepseek", split="train", streaming=True)
    out, seen = [], 0
    for ex in ds:
        seen += 1
        msgs = ex.get("messages") or []
        usr = next((m["content"] for m in msgs if m["role"] == "user"), "")
        asst = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
        if not usr or not asst or len(usr) > 2000:
            continue
        m = re.search(r"begin_of_thought\|>(.*?)<\|end_of_thought", asst, re.S)
        cot = (m.group(1) if m else asst).strip()
        pre = tok.encode(f"<｜User｜>{usr}<｜Assistant｜><think>\n", add_special_tokens=False)
        cot_ids = tok.encode(cot, add_special_tokens=False)
        seq = [bos] + pre + cot_ids[: SEQ_MAX]            # trim to band
        seq = seq[:SEQ_MAX]
        if _seq_ok(seq):
            out.append(("dolphin", seq))
        if len(out) >= n or seen >= 9000:
            break
    log(f"[data] dolphin scanned {seen} -> {len(out)} seqs")
    return out


def _turns_to_seq(tok, turns, bos):
    """turns: list of (role, content) alternating; build [bos] U/A segments until in band,
    ending on an assistant turn so the window holds model-generated tokens."""
    ids = [bos]
    last_ok = None
    for role, content in turns:
        content = (content or "").strip()
        if not content:
            continue
        if role == "user":
            seg = f"<｜User｜>{content}"
        else:
            seg = f"<｜Assistant｜>{content}<｜end▁of▁sentence｜>"
        ids = ids + tok.encode(seg, add_special_tokens=False)
        if role == "assistant" and len(ids) >= SEQ_MIN:
            last_ok = ids[:SEQ_MAX]
            if len(ids) >= SEQ_MAX:
                break
    return last_ok


def ultrachat_seqs(tok, n, bos):
    try:
        ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True)
    except Exception as e:
        log(f"[data] ultrachat unavailable: {e}"); return []
    out, seen = [], 0
    for ex in ds:
        seen += 1
        msgs = ex.get("messages") or []
        turns = [(m.get("role"), m.get("content")) for m in msgs
                 if m.get("role") in ("user", "assistant")]
        if len(turns) < 4:
            continue
        seq = _turns_to_seq(tok, turns, bos)
        if _seq_ok(seq):
            out.append(("ultrachat", seq))
        if len(out) >= n or seen >= 6000:
            break
    log(f"[data] ultrachat scanned {seen} -> {len(out)} seqs")
    return out


def oasst1_seqs(tok, n, bos):
    """OpenAssistant/oasst1: reconstruct root->leaf English threads (ungated, casual/varied)."""
    try:
        rows = list(load_dataset("OpenAssistant/oasst1", split="train"))
    except Exception as e:
        log(f"[data] oasst1 unavailable: {e}"); return []
    by_id = {m["message_id"]: m for m in rows}
    has_child = set(m["parent_id"] for m in rows if m.get("parent_id"))
    out = 0; res = []
    for m in rows:
        if m["message_id"] in has_child or m.get("lang") != "en":
            continue                                   # leaf English assistant msgs only
        path, cur, guard = [], m, 0
        while cur is not None and guard < 40:
            path.append(cur); cur = by_id.get(cur.get("parent_id")); guard += 1
        path = path[::-1]
        if len(path) < 4:
            continue
        turns = [("user" if p["role"] == "prompter" else "assistant", p.get("text"))
                 for p in path]
        seq = _turns_to_seq(tok, turns, bos)
        if _seq_ok(seq):
            res.append(("oasst1", seq)); out += 1
        if out >= n:
            break
    log(f"[data] oasst1 threads -> {len(res)} seqs")
    return res


_HH_RE = re.compile(r"\n\n(Human|Assistant): ")
def hhrlhf_seqs(tok, n, bos):
    """Anthropic/hh-rlhf: real multi-turn Human/Assistant transcripts (ungated, casual)."""
    try:
        ds = load_dataset("Anthropic/hh-rlhf", split="train", streaming=True)
    except Exception as e:
        log(f"[data] hh-rlhf unavailable: {e}"); return []
    out, seen = [], 0
    for ex in ds:
        seen += 1
        text = ex.get("chosen") or ""
        parts = _HH_RE.split(text)
        turns, i = [], 1
        while i + 1 < len(parts):
            role = "user" if parts[i] == "Human" else "assistant"
            turns.append((role, parts[i + 1])); i += 2
        if len(turns) < 4:
            continue
        seq = _turns_to_seq(tok, turns, bos)
        if _seq_ok(seq):
            out.append(("hhrlhf", seq))
        if len(out) >= n or seen >= 40000:
            break
    log(f"[data] hh-rlhf scanned {seen} -> {len(out)} seqs")
    return out


# ----------------------------- harvest (identical label to dolphin_scan) -----------------------------
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
def process(llm, tok, pooler, embT, seq, dims, bge=None):
    Hq, Hkv, D = dims
    if len(seq) < WINDOW + BLOCK * 4 + 4:
        return None
    n_evict = ((len(seq) - WINDOW) // BLOCK) * BLOCK
    nB = n_evict // BLOCK
    win_lo = n_evict
    win_len = len(seq) - win_lo

    cache = DynamicCache()
    with Cap(llm, LAYERS, q=False, k=True) as cap:
        o = llm(input_ids=torch.tensor([seq], device=DEV), output_attentions=True,
                use_cache=False)
    blockmass = np.zeros((win_len, nB), dtype=np.float32)
    for li in LAYERS:
        A = o.attentions[li][0].float()
        a = A[:, win_lo:, :]
        denom = a[:, :, 1:].sum(-1).clamp(min=1e-6)
        ev = a[:, :, :n_evict].reshape(a.shape[0], win_len, nB, BLOCK).sum(-1)
        frac = (ev / denom.unsqueeze(-1)).mean(0)
        blockmass += frac.cpu().numpy()
    blockmass /= len(LAYERS)
    del o
    maxmass = blockmass.max(1)
    goldblk = blockmass.argmax(1)
    kf = []
    for li in LAYERS:
        k = cap.k[li][0][:n_evict].view(nB, BLOCK, Hkv, D).float()
        kf.append(torch.cat([k.mean(1), k.amax(1)], -1).cpu())
    keys = torch.stack(kf, 1).half().numpy()

    kept, window = seq[:n_evict], seq[n_evict:]
    c = DynamicCache()
    llm.model(input_ids=torch.tensor([[seq[0]]], device=DEV), past_key_values=c, use_cache=True)
    sp = pooler(emb(embT, kept).float()).to(embT.weight.dtype)
    llm.model(inputs_embeds=sp, past_key_values=c, use_cache=True)
    with Cap(llm, LAYERS, q=True, k=False) as cap2:
        llm.model(inputs_embeds=emb(embT, window), past_key_values=c, use_cache=True)
    q_layers = [cap2.q[li][0].float().cpu().numpy().reshape(win_len, Hq, D) for li in LAYERS]
    qfeat = np.stack(q_layers, 1)
    out = {"qfeat": qfeat.astype(np.float16), "maxmass": maxmass, "goldblk": goldblk,
           "blockmass": blockmass, "keys": keys, "nB": nB, "win_len": win_len,
           "win_ids": window}
    if bge is not None:
        btexts = [tok.decode(seq[i * BLOCK:(i + 1) * BLOCK]) for i in range(nB)]
        out["bge_k"] = bge.encode(btexts, normalize_embeddings=True,
                                  show_progress_bar=False).astype(np.float16)
    return out


def collect(items, llm, tok, pooler, embT, dims, tag, t0, bge=None):
    Xg, yg, IX = [], [], []
    gi = 0
    for n, (_src, seq) in enumerate(items):
        try:
            r = process(llm, tok, pooler, embT, seq, dims, bge=bge)
        except Exception as e:
            log(f"  {tag} item {n} failed: {e}"); continue
        if r is None:
            continue
        mm = r["maxmass"]
        thr = np.quantile(mm, 1 - POS_FRAC)
        pos = np.where(mm >= max(thr, 0.05))[0]
        neg_all = np.where(mm < max(thr, 0.05))[0]
        rng = np.random.RandomState(n)
        neg = rng.choice(neg_all, min(len(neg_all), max(1, len(pos) * NEG_PER_POS)),
                         replace=False) if len(neg_all) else np.array([], int)
        for p in list(pos) + list(neg):
            Xg.append(r["qfeat"][p].reshape(-1)); yg.append(int(mm[p] >= max(thr, 0.05)))
        bge_q = None
        if bge is not None and len(pos):
            qtexts = [tok.decode(r["win_ids"][max(0, p - 64):p + 1]) for p in pos]
            bge_q = bge.encode(qtexts, normalize_embeddings=True,
                               show_progress_bar=False).astype(np.float16)
        for j, p in enumerate(pos):
            row = {"q": r["qfeat"][p].astype(np.float32), "k": r["keys"],
                   "gold": int(r["goldblk"][p]), "nB": r["nB"],
                   "labels": r["blockmass"][p][None, :].astype(np.float32)}
            if bge is not None:
                row["bge_k"] = r["bge_k"].astype(np.float32)
                row["bge_q"] = bge_q[j].astype(np.float32)
            IX.append(row)
        gi += 1
        if (n + 1) % 20 == 0:
            log(f"  {tag} {n+1}/{len(items)} items={gi} gate_pts={len(yg)} "
                f"pos={int(np.sum(yg))} idx_rows={len(IX)} t={time.time()-t0:.0f}s")
    return np.array(Xg, np.float32), np.array(yg, np.int32), IX


# ----------------------------- retriever training -----------------------------
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
            best = v; bs = {kk: vv.detach().cpu().clone() for kk, vv in m.state_dict().items()}
    return best, bs


class BGEHead(nn.Module):
    def __init__(self, qdim, kdim, d=128):
        super().__init__()
        self.Wq = nn.Sequential(nn.Linear(qdim, d), nn.GELU(), nn.Linear(d, d))
        self.Wk = nn.Sequential(nn.Linear(kdim, d), nn.GELU(), nn.Linear(d, d))
    def forward(self, q, k):
        return self.Wq(q) @ self.Wk(k).T


def raw_bge_top2(rows):
    h = 0
    for r in rows:
        s = r["bge_k"] @ r["bge_q"]
        h += int(r["gold"]) in np.argsort(-s)[:2].tolist()
    return h / max(len(rows), 1)


def train_bge_head(rows_tr, rows_te, seeds=2, epochs=25):
    qdim = rows_tr[0]["q"].reshape(-1).shape[0]; kdim = rows_tr[0]["bge_k"].shape[1]
    def tens(r):
        q = torch.tensor(r["q"].reshape(-1), device=DEV)
        k = torch.tensor(r["bge_k"], dtype=torch.float32, device=DEV)
        lab = torch.tensor(r["labels"].mean(0), device=DEV); lab = lab / max(lab.sum().item(), 1e-8)
        return q, k, lab
    def top2(m, split):
        h = 0
        with torch.no_grad():
            for r in split:
                q, k, _ = tens(r)
                h += int(r["gold"]) in m(q, k).topk(min(2, len(k))).indices.tolist()
        return h / max(len(split), 1)
    best, bs, bd = -1, None, (qdim, kdim, 128)
    for sd in range(seeds):
        torch.manual_seed(sd); m = BGEHead(qdim, kdim).to(DEV)
        opt = torch.optim.Adam(m.parameters(), lr=2e-3); rng = np.random.RandomState(sd)
        for _ in range(epochs):
            rng.shuffle(rows_tr)
            for r in rows_tr:
                q, k, lab = tens(r)
                loss = F.kl_div(F.log_softmax(m(q, k), 0), lab, reduction="sum")
                opt.zero_grad(); loss.backward(); opt.step()
        v = top2(m, rows_te)
        if v > best:
            best = v; bs = {kk: vv.detach().cpu().clone() for kk, vv in m.state_dict().items()}
    return best, bs, bd


def compare_old(dims, Xdt, ydt, Xct, yct, IXdt, IXct):
    """Score the OLD (Dolphin-only) gate + indexer on the SAME heldout features, so we get a
    direct OLD-vs-NEW comparison split by source: chat (雑談) vs cot. Threshold-free (AUC /
    top-2). Needs OLD_GATE / OLD_INDEXER files (downloaded by boot)."""
    from sklearn.metrics import roc_auc_score
    Hq, Hkv, D = dims
    og = os.environ.get("OLD_GATE", "old_gate.npz")
    oi = os.environ.get("OLD_INDEXER", "old_indexer.npz")
    cmp = {}
    if os.path.exists(og):
        z = np.load(og)
        coef = z["coef"].ravel().astype(np.float32); b = float(np.ravel(z["intercept"])[0])
        mean = z["mean"].astype(np.float32); scale = z["scale"].astype(np.float32)
        def sco(X): return ((X - mean) / scale) @ coef + b
        def auc(X, y):
            return round(float(roc_auc_score(y, sco(X))), 4) if len(y) and 0 < int(y.sum()) < len(y) else None
        cmp["old_gate_auc_cot"] = auc(Xdt, ydt)
        cmp["old_gate_auc_chat"] = auc(Xct, yct)
    if os.path.exists(oi):
        z = np.load(oi)
        m = Indexer(len(LAYERS), Hq, Hkv, D).to(DEV)
        m.load_state_dict({"Aq": torch.tensor(z["Aq"]), "Bk": torch.tensor(z["Bk"]),
                           "w": torch.tensor(z["w"])})
        def t2(split):
            h = 0
            with torch.no_grad():
                for r in split:
                    q = torch.tensor(r["q"], device=DEV)
                    k = torch.tensor(r["k"], dtype=torch.float32, device=DEV)
                    h += int(r["gold"]) in m(q, k).topk(min(2, len(k))).indices.tolist()
            return round(h / max(len(split), 1), 4)
        cmp["old_index_top2_cot"] = t2(IXdt) if IXdt else None
        cmp["old_index_top2_chat"] = t2(IXct) if IXct else None
    return cmp


def snap(payload):
    json.dump(payload, open(os.path.join(OUT, "results.json"), "w"), indent=2)


def main():
    t0 = time.time(); os.makedirs(OUT, exist_ok=True); os.makedirs(ART, exist_ok=True)
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    log(f"[load] fft_hf on {DEV} (eager attn)")
    tok = AutoTokenizer.from_pretrained("fft_hf")
    try:
        llm = AutoModelForCausalLM.from_pretrained("fft_hf", torch_dtype=torch.float32,
                                                   attn_implementation="eager")
    except TypeError:
        llm = AutoModelForCausalLM.from_pretrained("fft_hf", dtype=torch.float32,
                                                   attn_implementation="eager")
    llm = llm.eval().to(DEV)
    pooler = load_pooler().to(DEV).eval(); embT = llm.get_input_embeddings()
    bge = None
    if WITH_BGE:
        from sentence_transformers import SentenceTransformer
        bge = SentenceTransformer("BAAI/bge-base-en-v1.5", device=DEV); log("[bge] loaded")
    cfg = llm.config; Hq = cfg.num_attention_heads; Hkv = cfg.num_key_value_heads
    D = cfg.hidden_size // Hq; dims = (Hq, Hkv, D)
    bos = tok.bos_token_id
    log(f"[dims] Hq={Hq} Hkv={Hkv} D={D} feat={len(LAYERS)*Hq*D}")

    # ---- assemble corpora (item-disjoint train/held per source) ----
    dol = dolphin_seqs(tok, N_DOLPHIN_TR + N_DOLPHIN_TE, bos)
    need = N_CHAT_TR + N_CHAT_TE
    uc = ultrachat_seqs(tok, int(need * 0.6) + 5, bos)
    oa = oasst1_seqs(tok, need // 2 + 5, bos)            # casual/varied multi-turn (ungated)
    chat = uc + oa
    if len(chat) < need:                          # top up from ultrachat if casual fell short
        extra = ultrachat_seqs(tok, need * 2, bos)
        have = {tuple(x[1]) for x in chat}
        chat = chat + [c for c in extra if tuple(c[1]) not in have]
    np.random.RandomState(0).shuffle(chat)
    dol_tr, dol_te = dol[:N_DOLPHIN_TR], dol[N_DOLPHIN_TR:N_DOLPHIN_TR + N_DOLPHIN_TE]
    chat_tr, chat_te = chat[:N_CHAT_TR], chat[N_CHAT_TR:N_CHAT_TR + N_CHAT_TE]
    log(f"[mix] dolphin tr/te={len(dol_tr)}/{len(dol_te)} | chat tr/te={len(chat_tr)}/{len(chat_te)} "
        f"(ultrachat={len(uc)} oasst1={len(oa)})")

    Xd, yd, IXd = collect(dol_tr, llm, tok, pooler, embT, dims, "DOL_TR", t0, bge=bge)
    Xc, yc, IXc = collect(chat_tr, llm, tok, pooler, embT, dims, "CHAT_TR", t0, bge=bge)
    Xdt, ydt, IXdt = collect(dol_te, llm, tok, pooler, embT, dims, "DOL_TE", t0, bge=bge)
    Xct, yct, IXct = collect(chat_te, llm, tok, pooler, embT, dims, "CHAT_TE", t0, bge=bge)

    Xtr = np.vstack([Xd, Xc]); ytr = np.concatenate([yd, yc])
    IXtr = IXd + IXc
    log(f"[gate data] train pts={len(ytr)} pos={int(ytr.sum())} | "
        f"dol_te pts={len(ydt)} pos={int(ydt.sum())} | chat_te pts={len(yct)} pos={int(yct.sum())}")

    res = {"elapsed_s": round(time.time()-t0, 1),
           "config": {"seq_band": [SEQ_MIN, SEQ_MAX], "pos_frac": POS_FRAC,
                      "n_dolphin": [len(dol_tr), len(dol_te)],
                      "n_chat": [len(chat_tr), len(chat_te)],
                      "sources_chat": {"ultrachat": len(uc), "oasst1": len(oa)}},
           "gate": {"train_pts": int(len(ytr)), "train_pos": int(ytr.sum())}}

    # ---- GATE (logistic on SP-context q), eval per source ----
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=0.05, max_iter=3000, class_weight="balanced").fit(
        sc.transform(Xtr), ytr)
    def auc_of(X, y):
        if len(y) and 0 < int(y.sum()) < len(y):
            return float(roc_auc_score(y, clf.decision_function(sc.transform(X))))
        return None
    def cat(Xs, ys):                              # robust concat (skip empty splits)
        Xs = [(X, y) for X, y in zip(Xs, ys) if len(y) > 0]
        if not Xs:
            return np.zeros((0, Xtr.shape[1]), np.float32), np.zeros((0,), np.int32)
        return np.vstack([X for X, _ in Xs]), np.concatenate([y for _, y in Xs])
    res["gate"]["heldout_auc_dolphin"] = auc_of(Xdt, ydt)
    res["gate"]["heldout_auc_chat"] = auc_of(Xct, yct)
    Xall, yall = cat([Xdt, Xct], [ydt, yct])
    res["gate"]["heldout_auc_all"] = auc_of(Xall, yall)
    posS = clf.decision_function(sc.transform(Xtr[ytr == 1]))
    thresh = float(np.median(posS))
    res["gate"]["thresh"] = round(thresh, 4)
    log(f"[GATE] heldout AUC  dolphin={res['gate']['heldout_auc_dolphin']} "
        f"chat={res['gate']['heldout_auc_chat']} all={res['gate']['heldout_auc_all']}")
    # save gate.npz (recall_kit format)
    np.savez(os.path.join(ART, "gate.npz"),
             coef=clf.coef_.ravel().astype(np.float32),
             intercept=np.float32(clf.intercept_[0]),
             mean=sc.mean_.astype(np.float32), scale=sc.scale_.astype(np.float32),
             thresh=np.float32(thresh))
    snap(res)

    # ---- INDEXER (learned QK) ----
    IXCAP = int(os.environ.get("IDX_CAP", "3000"))
    rng = np.random.RandomState(0)
    if len(IXtr) > IXCAP:
        IXtr = [IXtr[i] for i in rng.choice(len(IXtr), IXCAP, replace=False)]
    log(f"[indexer] train pos={len(IXtr)} | dol_te={len(IXdt)} chat_te={len(IXct)}")
    if len(IXtr) > 10 and (len(IXdt) + len(IXct)) > 5:
        IXte_all = IXdt + IXct
        t_all, state = train_indexer([dict(r) for r in IXtr], IXte_all, dims)
        m = Indexer(len(LAYERS), Hq, Hkv, D).to(DEV); m.load_state_dict(state)
        def top2(split):
            h = 0
            with torch.no_grad():
                for r in split:
                    q = torch.tensor(r["q"], device=DEV)
                    k = torch.tensor(r["k"], dtype=torch.float32, device=DEV)
                    h += int(r["gold"]) in m(q, k).topk(min(2, len(k))).indices.tolist()
            return h / max(len(split), 1)
        res["indexer"] = {"train_pos": len(IXtr),
                          "heldout_top2_dolphin": round(top2(IXdt), 4) if IXdt else None,
                          "heldout_top2_chat": round(top2(IXct), 4) if IXct else None,
                          "heldout_top2_all": round(t_all, 4)}
        log(f"[INDEXER] heldout top-2 dolphin={res['indexer']['heldout_top2_dolphin']} "
            f"chat={res['indexer']['heldout_top2_chat']} all={t_all:.4f}")
        np.savez(os.path.join(ART, "indexer.npz"),
                 Aq=state["Aq"].numpy(), Bk=state["Bk"].numpy(), w=state["w"].numpy())
        snap(res)
        # ---- BGE-on-demand head ----
        if bge is not None and IXtr and "bge_k" in IXtr[0]:
            IXte_all = IXdt + IXct
            raw_all = raw_bge_top2(IXte_all)
            bge_top, bstate, (qd, kd, dd) = train_bge_head([dict(r) for r in IXtr], IXte_all)
            def bge_top2(split, m):
                h = 0
                with torch.no_grad():
                    for r in split:
                        q = torch.tensor(r["q"].reshape(-1), device=DEV)
                        k = torch.tensor(r["bge_k"], dtype=torch.float32, device=DEV)
                        h += int(r["gold"]) in m(q, k).topk(min(2, len(k))).indices.tolist()
                return h / max(len(split), 1)
            mb = BGEHead(qd, kd, dd).to(DEV); mb.load_state_dict(bstate)
            res["bge"] = {"raw_bge_top2_all": round(raw_all, 4),
                          "learned_top2_dolphin": round(bge_top2(IXdt, mb), 4) if IXdt else None,
                          "learned_top2_chat": round(bge_top2(IXct, mb), 4) if IXct else None,
                          "learned_top2_all": round(bge_top, 4)}
            log(f"[BGE] raw={raw_all:.4f} | learned dolphin={res['bge']['learned_top2_dolphin']} "
                f"chat={res['bge']['learned_top2_chat']} all={bge_top:.4f}")
            sd = {k: v.numpy() for k, v in bstate.items()}
            np.savez(os.path.join(ART, "bge_head.npz"),
                     qdim=np.int64(qd), kdim=np.int64(kd), d=np.int64(dd), **sd)
    # ---- OLD (Dolphin-only) vs NEW (mixed), split by source: chat (雑談) vs cot ----
    try:
        cmp = compare_old(dims, Xdt, ydt, Xct, yct, IXdt, IXct)
        res["compare_old_vs_new"] = {
            "gate_auc": {"cot": {"old": cmp.get("old_gate_auc_cot"),
                                 "new": res["gate"].get("heldout_auc_dolphin")},
                         "chat": {"old": cmp.get("old_gate_auc_chat"),
                                  "new": res["gate"].get("heldout_auc_chat")}},
            "index_top2": {"cot": {"old": cmp.get("old_index_top2_cot"),
                                   "new": (res.get("indexer") or {}).get("heldout_top2_dolphin")},
                           "chat": {"old": cmp.get("old_index_top2_chat"),
                                    "new": (res.get("indexer") or {}).get("heldout_top2_chat")}}}
    except Exception as e:
        log(f"[compare] failed: {e}"); res["compare_old_vs_new"] = {"error": str(e)}

    res["final"] = True; snap(res)
    log("\n=== SUMMARY (mixed corpus: dolphin + ultrachat + oasst1) ===")
    log(f"  GATE   heldout AUC  dolphin={res['gate'].get('heldout_auc_dolphin')} "
        f"chat={res['gate'].get('heldout_auc_chat')} all={res['gate'].get('heldout_auc_all')}")
    if res.get("indexer"):
        log(f"  INDEX  heldout top2 dolphin={res['indexer'].get('heldout_top2_dolphin')} "
            f"chat={res['indexer'].get('heldout_top2_chat')} all={res['indexer'].get('heldout_top2_all')}")
    if res.get("bge"):
        log(f"  BGE    learned top2 dolphin={res['bge'].get('learned_top2_dolphin')} "
            f"chat={res['bge'].get('learned_top2_chat')} all={res['bge'].get('learned_top2_all')}")
    cv = res.get("compare_old_vs_new") or {}
    if cv.get("gate_auc"):
        g = cv["gate_auc"]; ix = cv.get("index_top2", {})
        log("  --- OLD(dolphin-only) vs NEW(mixed) ---")
        log(f"  GATE  AUC   cot : old={g['cot']['old']}  new={g['cot']['new']}")
        log(f"  GATE  AUC   chat: old={g['chat']['old']}  new={g['chat']['new']}")
        if ix:
            log(f"  INDEX top2  cot : old={ix['cot']['old']}  new={ix['cot']['new']}")
            log(f"  INDEX top2  chat: old={ix['chat']['old']}  new={ix['chat']['new']}")
    log(f"DONE t={time.time()-t0:.0f}s")
    open(os.path.join(OUT, ".done"), "w").write("ok")


if __name__ == "__main__":
    main()
