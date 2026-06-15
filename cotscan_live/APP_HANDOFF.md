# APP_HANDOFF — on-device recall (SP compression + gate + BGE-on-demand)

Read this and you can implement the recall system in the app. It is the **on-device** target:
across turns you persist **token IDs only**; a per-position **GATE** detects when the model is
about to need a fact that scrolled out of the exposure window, and a **BGE-on-demand RETRIEVER**
pulls the right evicted 128-token block back in (verbatim) so the value survives SP compression.

HF repo: **`baya1116/hypernet-sp-distill`**. Resolve URL pattern:
`https://huggingface.co/baya1116/hypernet-sp-distill/resolve/main/<path>`

---

## 1. Artifacts and exact locations (ありか)

| what | path in repo | format |
|---|---|---|
| **GATE (use this)** | `trigger_experiment/ondevice_recall/gate.npz` | npz, see §4 |
| **BGE bridge head (use this)** | `trigger_experiment/ondevice_recall/bge_head.npz` | npz, see §5 |
| **SP pooler** | `trigger_experiment/ondevice_recall/pooler.pt` (=`fft_out/pooler.pt`) | torch state, AttnPoolSP |
| base LLM (HF format) | `recall_runtime/fft_hf/` (`model.safetensors`, tokenizer, config) | DeepSeek-R1-Distill-Qwen-1.5B, FFT student |
| base LLM (raw build) | `fft_out/student.pt` + `build_fft_hf.py` | builds `fft_hf/` |
| pooler loader | `hypernet_sp/attn_export3_torch.py` → `load_pooler()` | defines AttnPoolSP |
| reference runtime/retriever | `recall_runtime/recall_kit/{runtime,retriever,gate}.py` | Python reference impl |
| reference block archive (IDs→keys) | `hypernet_sp/block_recall.py` (`BlockArchive`) | raw-QK variant (server) |
| end-to-end eval (canonical loop) | git `cotscan_live/recall_gen.py` (`reply()`, `retrieve_bge()`) | the exact loop to port |
| external dep | `BAAI/bge-base-en-v1.5` (sentence-transformers / convertible to CoreML/MLX) | 110M encoder |

Direct links:
- gate: https://huggingface.co/baya1116/hypernet-sp-distill/blob/main/trigger_experiment/ondevice_recall/gate.npz
- bge_head: https://huggingface.co/baya1116/hypernet-sp-distill/blob/main/trigger_experiment/ondevice_recall/bge_head.npz
- pooler: https://huggingface.co/baya1116/hypernet-sp-distill/blob/main/trigger_experiment/ondevice_recall/pooler.pt
- base model dir: https://huggingface.co/baya1116/hypernet-sp-distill/tree/main/recall_runtime/fft_hf
- recall_kit (reference): https://huggingface.co/baya1116/hypernet-sp-distill/tree/main/recall_runtime/recall_kit
- full status / results: git branch `claude/hypernet-sp-spec-review-qrafn9`,
  `cotscan_live/STATUS_2026-06-14_recall_improvements.md`

---

## 2. Constants (1.5B model)

```
hidden = 1536,  Hq = 12 (attn heads),  Hkv = 2 (KV heads),  D = 128 (head dim)
LAYERS   = (8, 14, 20)      # the 3 layers the gate + retriever read
RW       = 512              # exposure window: last RW tokens stay verbatim
BLOCK    = 128              # evicted tokens are sealed into 128-token blocks
C        = 64               # generation chunk between rebuilds
RECALL_K = 2                # inject top-2 blocks on a fire
FEATURE  = len(LAYERS) * Hq * D = 3*12*128 = 4608     # gate / bridge query dim
BOS / EOS / <｜User｜> / <｜Assistant｜>  : DeepSeek-R1 chat special tokens
```

---

## 3. The runtime loop (what to implement)

State you persist across turns = **token IDs of the conversation only**. Everything else is
transient. Generation runs in chunks of `C` tokens; before each chunk you "rebuild" the prefix.

```
gen      = [all tokens so far: prior turns + this turn's forced user tokens]
kept     = []          # evicted tokens already folded into the SP gist
absorbed = 0
archive  = []          # list of {ids:[128]} sealed blocks  (IDs ONLY — no vectors)
rec_emb  = None        # currently-injected retrieved block embeddings (None until a fire)

loop until end-of-turn:
    R  = min(len(gen), RW)          # window length (verbatim tail)
    nd = len(gen) - R               # everything before the window is "evicted"
    if nd > absorbed:               # newly evicted tokens
        archive.extend(gen[absorbed:nd])     # seal into 128-tok blocks (store IDs)
        kept += gen[absorbed:nd];  absorbed = nd

    # ---- GATE (per rebuild) ----
    if archive not empty:
        gq = pre-RoPE q_proj output of the LAST token at layers 8/14/20   # captured in prev fwd
        score = ((concat(gq) - mean) / scale) · coef + intercept          # §4
        if score > THRESH:
            rid = bge_retrieve(archive, gq, k=RECALL_K)                   # §5
            if rid: rec_emb = embed(rid)

    # ---- build prefix and run the model ----
    prefix = [BOS]
           + [ SP = pooler(embed(kept)) ]          # SP gist of evicted history (if kept)
           + ([ rec_emb ] if rec_emb is not None)   # injected evicted block(s), verbatim emb
           + [ embed(gen[len(gen)-R : ]) ]          # the verbatim window
    logits = model(prefix)   # also capture gq (last-token q_proj at 8/14/20) for next gate check

    generate C tokens greedily, append to gen   (stop on EOS or cap)
```

Notes:
- The gate score uses the q of the **current last token** under the **SP-compressed** prefix —
  that's the exact feature it was trained on. Capture `q_proj` output (pre-RoPE) of the last
  position at layers 8/14/20; concat in layer order → 4608-vector.
- Once `rec_emb` is set it stays injected for the rest of the turn (re-evaluate per turn).
- `pooler` = AttnPoolSP (`load_pooler("pooler.pt")`); it maps the embedded `kept` tokens to a
  short SP gist. Port from `hypernet_sp/attn_export3_torch.py`.

---

## 4. GATE — `gate.npz` (logistic, per-position recall detector)

Keys (all the model needs):
```
coef       float32 [4608]
intercept  float32 scalar      (= -6.2547)
mean       float32 [4608]
scale      float32 [4608]
thresh     float32 scalar      (= -2.8222, the 90%-held-recall point)
```
Scoring (q = the 4608 gate feature from §3):
```
score = ((q - mean) / scale) · coef + intercept
fire  = score > THRESH
```
**Operating threshold for generation**: use **THRESH ≈ −1.5 to −2.0** (recall-favoring; −2.0
gave 88% chat recall). The baked-in `thresh` (−2.82) is the held-out 90%-recall point. Lower =
more recall + more false fires; higher = fewer fires + less recall. See §7 for the trade.

---

## 5. RETRIEVER — BGE-on-demand (`bge_head.npz` + BAAI/bge-base-en-v1.5)

**State-free**: blocks are stored as token IDs only. At a fire you BGE-encode the candidate
blocks' decoded text on the fly. The query is the gate's hidden q (free — already computed).

`bge_head.npz` (a 2-tower bridge; PyTorch `Sequential(Linear, GELU, Linear)` each):
```
qdim=4608, kdim=768, d=128
Wq.0.weight [128,4608]  Wq.0.bias [128]   Wq.2.weight [128,128]  Wq.2.bias [128]
Wk.0.weight [128,768]   Wk.0.bias [128]   Wk.2.weight [128,128]  Wk.2.bias [128]
Wq(x) = Linear2(GELU(Linear0(x)))   # 4608 -> 128
Wk(x) = Linear2(GELU(Linear0(x)))   # 768  -> 128
```
Retrieval:
```
def bge_retrieve(archive, gq, k=2):
    cands = archive.blocks (+ pending buffer if >=16 toks)     # each is token IDs
    q  = concat(gq)                      # RAW 4608 vector (NOT standardized — bge_head wants raw)
    qz = Wq(q)                           # [128]
    for each block:
        text  = tokenizer.decode(block.ids)
        kemb  = BGE.encode(text, normalize=True)   # [768]   (BAAI/bge-base-en-v1.5)
        kz    = Wk(kemb)                            # [128]
        score = qz · kz
    top = top-k blocks by score, in chronological order
    return concat(top.ids)               # inject these token IDs (embedded) in §3
```
Important: **raw BGE cosine is useless (top-2 0.29)** — the bridge head is mandatory (0.9997
held-out, and == raw-QK in generation). Query side is the LLM hidden q, NOT a BGE of the query.

---

## 6. Measured (so you know the target)

- Multi-fact chat needle (N=8, 24 questions, item-disjoint): **recall 88%** with recall on,
  **0%** without (SP-only). **BGE-on-demand == raw-QK, bit-identical** (both 21/24).
- Gate detection AUC: cot ≈ 0.90, chat ≈ 0.93 (maintained vs Dolphin-only chat 0.888).
- Retrieval (held-out): BGE+bridge top-2 0.9997; learned QK indexer 0.998 (but worse in
  generation, 75%); raw BGE cosine 0.29 (don't use).

---

## 7. On-device notes

- **Base 1.5B** is the only heavy part → 4-bit quant (MLX / llama.cpp-Metal / CoreML). MLX
  scaffolding exists: `cotscan_live/` and `*_mlx.py` (pooler_mlx, sp_mlx).
- **GATE** = one 4608-dim dot product per rebuild. Negligible.
- **BGE** = BAAI/bge-base-en-v1.5 (~110M), convert to CoreML/MLX; 4-bit ≈ 30–60 MB. Runs **only
  on a fire**, encoding a few short blocks → tens of ms. ~15–25× cheaper than recomputing the
  1.5B's own keys, which is why BGE-on-demand is the on-device choice.
- **Persisted state = token IDs only.** No KV / key vectors kept across turns.
- **Open item (don't be surprised)**: per-turn casual false-fire in generation is ~62% at 88%
  recall (held-out per-position is 3–5%). It is a recall/precision trade, best fixed at runtime,
  not with more gate data: fire-once-per-turn, hysteresis (require N consecutive fires before
  injecting), or a relevance gate on the top retrieval score before injecting. Or pick
  THRESH ≈ −1.2 for 75% recall / 38% false-fire if precision matters more than recall.

---

## 8. Minimal port checklist

1. Load `fft_hf` (4-bit) + tokenizer; expose hidden states / `q_proj` outputs at layers 8/14/20.
2. Port `AttnPoolSP` (pooler.pt) → SP gist of `kept`.
3. Implement the §3 chunked SP loop (window RW=512, blocks of 128, inject between SP and window).
4. Gate: load `gate.npz`, score per rebuild, fire at THRESH≈−1.5…−2.
5. BGE-on-demand: bundle `BAAI/bge-base-en-v1.5` (CoreML/MLX) + `bge_head.npz`; retrieve top-2 on
   fire; inject block IDs.
6. Keep only conversation token IDs across turns.

Reference implementation to mirror exactly: git `cotscan_live/recall_gen.py` → `reply()` (the
loop) and `retrieve_bge()` (the retriever), plus `recall_runtime/recall_kit/runtime.py`.
