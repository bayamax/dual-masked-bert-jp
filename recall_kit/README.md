# recall_kit — per-position recall for SP-compressed, infinite-turn memory

A small, trained-and-validated kit for the COTSCAN problem: during long chain-of-thought,
detect the rare moment the model needs a fact that has scrolled **past the 512-token
exposure window**, and pull the right evicted block back — without growing any per-turn
vector store.

Built and validated on **real DeepSeek-R1 / Dolphin-R1 long CoTs** (≥1000-token reasoning),
labelled by where full-KV attention naturally flies back past the window.

## Components

| module | what it does | persisted state |
|---|---|---|
| `RecallGate` | logistic on the live pre-RoPE query (layers 8/14/20, 4608-d) → "does this position need an evicted fact?" | none (reads the live hidden state) |
| `QKIndexer` | learned indexer in the model's **own QK space** → which evicted block | a ~3KB pooled key summary per block (harvested free during generation) |
| `BGERetriever` | **state-free**: only token ids persist; on a fire, BGE-embeds candidate block *texts* on demand and a tiny learned transform bridges them to the live query | none (ids only) |
| `RecallRuntime` | wiring: `prepare(seq)` → features; `decide(ctx, p, backend)` → gate + retrieve | — |

## Headline results (heldout, unseen items)

Detection (`RecallGate`), heldout **AUC = 0.958**.

Retrieval top-2 (which block), heldout:

| method | top-2 | persisted vectors |
|---|---|---|
| raw-QK (no learning) | 0.807 | – |
| **learned QK indexer** | **0.9995** | ~3 KB / block |
| raw-BGE cosine (no learning) | 0.286 | ids only |
| **learned transform on BGE** | **0.9997** | **ids only** |

The learned transform is the whole game: it lifts BGE retrieval from **0.286 → 0.9997**,
i.e. the **elegant, state-free path matches the QK-stored path** — only token ids ever
accumulate, and a light BGE + fixed head run only on the (rare) recall.

## Usage

```python
from recall_kit import RecallGate, QKIndexer, BGERetriever, RecallRuntime
gate = RecallGate.load("artifacts/gate.npz")
qk   = QKIndexer.load("artifacts/indexer.npz", dims=(Hq, Hkv, D), device="cuda")
bge  = BGERetriever.load("artifacts/bge_head.npz", sentence_transformer, device="cuda")

rt  = RecallRuntime(model, tok, pooler, gate, qk=qk, bge=bge, device="cuda")
ctx = rt.prepare(seq_ids)                      # [BOS][SP(evicted)][window]
for p in range(ctx.win_len):
    d = rt.decide(ctx, p, backend="bge")       # gate -> (only on fire) retrieve top block
    if d["fired"]:
        inject(ctx.block_ids[d["blocks"][0]])  # re-inject the recalled block verbatim
```

## Validation

- **Unit tests** (`build_package.py`): each component loads from its saved weights and is
  asserted on heldout fixtures — gate separation AUC, QK top-2, BGE-head top-2, BGE encode.
- **Composite test** (`composite_test.py`): the wired `RecallRuntime` runs end-to-end on
  unseen held items; reports gate precision/recall/F1, retrieval@true-recall, and the
  end-to-end "fired-on-a-real-recall-and-pulled-the-right-block" rate, for both backends.

Artifacts (weights + reports) and these results live in the HF repo under
`trigger_experiment/recall_kit/`.
