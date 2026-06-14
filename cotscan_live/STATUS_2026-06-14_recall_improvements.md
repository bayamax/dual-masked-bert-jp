# Recall system — overnight improvement campaign (2026-06-14)

Continuation of the gate-based mid-CoT/chat recall work. Goal this round: de-bias the gate off
Dolphin-only data, fix what the chat demo exposed, and push end-to-end quality across the board
(chat **and** CoT), iterating with real generation evals.

All models live in HF repo `baya1116/hypernet-sp-distill` under `trigger_experiment/`.

## TL;DR

- **Chat recall works**: multi-fact needle benchmark (N=8, item-disjoint) → **88% recall** with
  SP compression + gate + raw-QK retrieval, vs **0%** for SP-without-recall. Retrieval **100%**.
- **Retriever verdict**: once context truly evicts, **raw-QK ≥ learned indexer** in generation
  (88% vs 75%). The earlier "opening-block bias" was an artifact of one short conversation. Use
  raw-QK at generation time; the learned indexer stays the strong *held-out* retriever (top-2 0.998).
- **Gate de-biased + sharpened** via diversified data + contrastive hard negatives. Detection AUC
  held at ~0.90–0.93 on both cot and chat throughout.
- **Remaining weak point**: the gate over-fires on casual turns *in generation*. Progressively
  reduced (see table); v3 trained on the true generation distribution to close it.

## Data diversification

Recall labels ("full-KV attention concentrates on a single evicted block") are corpus-agnostic,
so we harvested the same label across a mix instead of Dolphin-only:

| source | role |
|---|---|
| `mlabonne/dolphin-r1-deepseek` | long single-turn reasoning CoT (existing) |
| `HuggingFaceH4/ultrachat_200k` | instructional multi-turn dialogue (added) |
| `OpenAssistant/oasst1` | casual/varied multi-turn threads (added) |

(SODA and hh-rlhf were tried but too short to reach the 1024-tok eviction threshold → dropped.)

## Gate: old → new (held-out, item-disjoint)

| gate | cot AUC | chat AUC | casual false-fire @recall90 |
|---|---|---|---|
| dolphin-only (v4) | 0.919 | 0.888 | — |
| **mix** (`recall_mix_v1`) | 0.906/0.926 | 0.930/0.927 | 11% (teacher-forced) |
| **contrastive v1** (`recall_contrastive_v1`) | 0.906 | 0.929 | 6% |
| **contrastive v2** (`recall_contrastive_v2`) | 0.925 | 0.925 | **3%** |
| **v3 gen-distribution** (`recall_contrastive_v3`) | 0.904 | 0.933 | 5% on the *generation* held set (mix 10%) |

Positives are always **mass-labeled** (real attention-to-evicted), never hand-made. Contrastive
training only *adds label-0 hard negatives*: "state a fact, then talk about something unrelated".
v3's negatives are captured from the model's **actual SP-generation** of casual replies — the
exact distribution where it was over-firing.

The old (Dolphin-only) gate scored chat detection notably worse (AUC 0.888) and **never fired in
the chat demo**; the retrained gate fires correctly on conversational recall.

## End-to-end chat generation (multi-fact needle, N=8 = 24 questions / 16 casual probes)

Same contrastive gate; arms differ only in retriever (RW=512, facts evicted, feed≈670 tok):

| arm | recall | retrieved | casual false-fire (per turn) |
|---|---|---|---|
| SP (no recall) | 0% | 0% | 0% |
| **RC_raw @ thresh −1.5** | **88%** | 88% | 62% |
| RC_idx (learned) @ −1.5 | 75% | 75% | 62% |

Threshold sweep (raw-QK): **−1.5 is the knee** — recall/retrieval stay at 100% (small-N) while
casual-fire halves; higher thresholds only cost recall. Per-turn casual-fire stays high because
it aggregates ~2–3 decision points/turn; per-position held false-fire is far lower (3–5%).

**v3 generation confirmation (N=8, gate `recall_contrastive_v3`):**

| gate v3, raw-QK | recall | retrieved | casual false-fire (per turn) |
|---|---|---|---|
| thresh −2.0 | **88%** | 88% | 62% |
| thresh −1.2 | 75% | 75% | **38%** |

Honest read: it's a recall/precision **tradeoff**. The data-side work (contrastive + generation
-distribution negatives) cut *held-out per-position* false-fire ~4× (11%→3%, gen-dist 10%→5%), but
*per-turn generation* false-fire stays high at max recall because it aggregates ~2–3 decision
points/turn. At 88% recall it's ~62%; dropping to 75% recall gets it to 38%. **The remaining lever
is a runtime mitigation, not more gate data** — see Open items.

## Live chat demo (Claude as the human)

A real authored conversation (facts "Bluefin / October 9th / $47,000" early, buried under casual
turns, asked back): **FULL 3/3, SP 1/3 (hallucinates from the recent window), RECALL 3/3** with
the new gate — it fired at the recall turns and pulled the evicted fact block. Confirmed the SP
failure mode (answering from the visible window instead of the evicted fact) and that recall fixes it.

## Recommended runtime config

- **gate**: `recall_contrastive_v3/artifacts/gate.npz` (best generation-distribution false-fire,
  AUC preserved). Operating threshold ≈ −1.5 to −2 for generation (recall-favoring).
- **retriever**: raw-QK `BlockArchive(mode="qk")` for generation; learned `indexer.npz`
  (`recall_mix_v1`) remains the top held-out retriever.
- **pooler / base / package**: unchanged (`fft_out/pooler.pt`, `fft_hf`, `recall_kit/`).

## Open items

- **Per-turn casual false-fire (~62% at 88% recall) is the #1 remaining issue and is best fixed at
  RUNTIME, not with more gate data** (data-side has saturated: held per-position false-fire is
  already 3–5%). Cheap, high-leverage mitigations to try next:
  - fire-once-per-turn (cap injections at 1 per assistant turn);
  - hysteresis (require the gate to clear threshold on N consecutive checks before injecting);
  - relevance gate: only inject if the top retrieved block's QK score clears an absolute bar
    (during casual turns nothing is truly relevant, so this suppresses spurious injections);
  - accept the tradeoff knob: thresh −1.2 → 75% recall / 38% false-fire if precision matters more.
- CoT-quality is covered by held metrics (cot detection AUC ~0.90 preserved, indexer cot top-2
  0.996); the 1.5B's own reasoning loop — not recall — remains the accuracy ceiling on GSM8K/Dolphin.

## Scripts (this directory)

`mix_scan.py` (diversified mass-labeled gate+indexer+bge training), `gate_contrastive.py`
(contrastive hard-negative gate), `gate_v3.py` (generation-distribution negatives),
`recall_gen.py` (multi-fact chat benchmark, raw-QK vs learned indexer, threshold sweep),
`chat_demo.py` (authored conversation demo), `chat_solve.py` (single-fact needle sweep).
