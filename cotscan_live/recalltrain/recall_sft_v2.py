"""recall_sft_v2.py — BODY FFT with the OFFICIAL gated recall (recall_kit RecallRuntime).

Fixes recall_sft.py on TWO counts:
  1. it injected the problem-HEAD block as a FAKE recall (no gate, no retrieval);
  2. it used LoRA, but the whole pipeline is FULL fine-tuning (the student fft_hf is an FFT).
Here every training example's recalled block is produced by the REAL mechanism
(RecallGate fires on a window position -> trained retriever pulls the top evicted block),
and the body is FULL fine-tuned (all params) to generate the window GIVEN that officially
retrieved block injected between SP and window — the exact inference distribution.
Recall components (gate/retriever) and the SP pooler are FROZEN; only the LLM body trains.

Env: N (examples), EP (epochs), BACKEND=bge|qk, LR (default 1e-5). Saves fft_recall_v3/ (full model).
"""
import os, time, random, numpy as np, torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from attn_export3_torch import load_pooler
from recall_kit import RecallGate, BGERetriever, QKIndexer, RecallRuntime

DEV = "cuda" if torch.cuda.is_available() else "cpu"
HERE = os.path.dirname(os.path.abspath(__file__))
N = int(os.environ.get("N", "2000")); EP = int(os.environ.get("EP", "1"))
BACKEND = os.environ.get("BACKEND", "bge"); LR = float(os.environ.get("LR", "1e-5"))
WINDOW, BLOCK = 512, 128
TAG_OPEN = '(Recalled from earlier: "'; TAG_CLOSE = '")'
def log(*a): print(*a, flush=True)


def main():
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained("fft_hf")
    try: llm = AutoModelForCausalLM.from_pretrained("fft_hf", torch_dtype=torch.float32).eval()
    except TypeError: llm = AutoModelForCausalLM.from_pretrained("fft_hf", dtype=torch.float32).eval()
    llm = llm.to(DEV)
    pooler = load_pooler("pooler.pt").to(DEV).eval()
    for p in pooler.parameters(): p.requires_grad = False
    gate = RecallGate.load("components/gate.npz")
    dims = (llm.config.num_attention_heads, llm.config.num_key_value_heads,
            llm.config.hidden_size // llm.config.num_attention_heads)
    if BACKEND == "bge":
        from sentence_transformers import SentenceTransformer
        bge = SentenceTransformer("BAAI/bge-small-en-v1.5", device=DEV)
        retr = BGERetriever.load("components/bge_head.npz", bge, device=DEV)
        rt = RecallRuntime(llm, tok, pooler, gate, bge=retr, device=DEV)
    else:
        qk = QKIndexer.load("components/indexer.npz", dims, device=DEV)
        rt = RecallRuntime(llm, tok, pooler, gate, qk=qk, device=DEV)
    log(f"[built {time.time()-t0:.0f}s] BACKEND={BACKEND} N={N} EP={EP}")

    # ---- build examples through the OFFICIAL recall (gate fired -> retrieved evicted block) ----
    ds = load_dataset("mlabonne/dolphin-r1-deepseek", split="train", streaming=True)
    data = []; seen = 0; fired_ct = 0
    for r in ds:
        if len(data) >= N: break
        msgs = r.get("messages") or []
        u = next((x.get("content") for x in msgs if x.get("role") == "user"), None)
        a = next((x.get("content") for x in msgs if x.get("role") == "assistant"), None)
        if not (u and a): continue
        seq = [tok.bos_token_id] + tok.encode(f"<｜User｜>{u}<｜Assistant｜>{a}", add_special_tokens=False)
        if len(seq) < WINDOW + BLOCK + 64: continue          # need real eviction + a window
        seen += 1
        ctx = rt.prepare(seq)
        if ctx is None: continue
        hit = None
        for p in range(ctx.win_len):                          # first fired position drives recall
            d = rt.decide(ctx, p, backend=BACKEND, topk=1)
            if d["fired"]: hit = d["blocks"][0]; break
        if hit is None: continue                              # no recall needed -> not a recall example
        fired_ct += 1
        block = ctx.block_ids[hit]
        n_evict = ((len(seq) - WINDOW) // BLOCK) * BLOCK
        kept, window = seq[:n_evict], seq[n_evict:]
        rb = tok.encode(TAG_OPEN, add_special_tokens=False) + block + tok.encode(TAG_CLOSE, add_special_tokens=False)
        data.append((kept, rb, window))
        if seen % 50 == 0: log(f"  scanned {seen} fired {fired_ct} kept {len(data)} t={time.time()-t0:.0f}s")
    log(f"[data] {len(data)} recall examples (scanned {seen}, gate fired on {fired_ct})")
    if not data: log("NO DATA — gate never fired?"); return

    # ---- FULL body fine-tune: predict the window given [SP][official recalled block][window] ----
    for p in llm.parameters(): p.requires_grad = True      # FFT (not LoRA) — matches the pipeline
    llm.config.use_cache = False
    try: llm.gradient_checkpointing_enable()               # keep activation memory bounded
    except Exception: pass
    embT = llm.get_input_embeddings()
    n_tr = sum(p.numel() for p in llm.parameters() if p.requires_grad)
    log(f"[FFT] trainable params = {n_tr/1e6:.0f}M (full body), lr={LR}")
    opt = torch.optim.AdamW([p for p in llm.parameters() if p.requires_grad], lr=LR)
    llm.train()
    for ep in range(EP):
        random.Random(ep).shuffle(data); tot = 0.0; nb = 0
        for kept, rb, window in data:
            with torch.no_grad():
                sp = pooler(embT(torch.tensor([kept], device=DEV)).float()).to(embT.weight.dtype)
            rec = embT(torch.tensor([rb], device=DEV))
            win = embT(torch.tensor([window], device=DEV))
            out = llm(inputs_embeds=torch.cat([sp, rec, win], 1))
            nctx = sp.shape[1] + rec.shape[1]; wl = len(window)
            lg = out.logits[0, nctx:nctx + wl - 1, :]
            loss = F.cross_entropy(lg.float(), torch.tensor(window[1:], device=DEV))
            loss.backward(); torch.nn.utils.clip_grad_norm_([p for p in llm.parameters() if p.requires_grad], 1.0)
            opt.step(); opt.zero_grad(); tot += loss.item(); nb += 1
            if nb % 50 == 0: log(f"  ep{ep+1} {nb}/{len(data)} loss={tot/nb:.4f} t={time.time()-t0:.0f}s")
        log(f"[ep {ep+1}/{EP}] loss={tot/max(nb,1):.4f} t={time.time()-t0:.0f}s")
    llm.save_pretrained(os.path.join(HERE, "fft_recall_v3"))      # full fine-tuned model
    tok.save_pretrained(os.path.join(HERE, "fft_recall_v3"))
    open(os.path.join(HERE, ".done"), "w").write("ok"); log(f"TRAIN_DONE t={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
