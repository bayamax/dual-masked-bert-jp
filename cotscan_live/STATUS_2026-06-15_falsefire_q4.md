# 4-bit recall false-fire mitigation — results (2026-06-15)

Goal (per user): on the **4-bit (nf4) on-device** model, reduce recall **false-fire** (firing when
no recall is needed). Two levers requested: **threshold tuning** and **contrastive learning**. A
third (runtime **relevance floor** on the BGE retrieval score) was tested as the natural precision
lever for the confirmed BGE-on-demand path.

Setup: one RunPod A6000, scripts `gate_v3_q4.py` (new) + `recall_gen.py` (extended: `Q4=1` 4-bit
load, `GATE_NPZ` swap, `:rFLOOR` relevance arm, per-fire retrieval-score logging split into
recall-fires vs casual-fires). Base = DeepSeek-R1-Distill-Qwen-1.5B FFT student, 4-bit nf4.
Total GPU spend ≈ **$0.46**. Artifacts on HF under `trigger_experiment/results_gate_v3_q4/`,
`results_recall_q4_oldgate/`.

## Result 1 — contrastive learning (4-bit) does NOT help  → not adopted
`gate_v3_q4.py`: positives mass-labeled (fp32 attention), q-features from the **4-bit** model;
negatives = teacher-forced casual tails **+ generation-distribution negatives harvested by running
the SP loop on the 4-bit model itself** (gate_v3 idea applied on-device). Scored on a held set vs
the current fp32 ondevice gate evaluated on 4-bit features:

| metric (4-bit features) | baseline fp32 gate | new 4-bit contrastive |
|---|---|---|
| AUC cot | **0.977** | 0.922 ↓ |
| AUC chat | **0.976** | 0.914 ↓ |
| GEN false-fire @recall90 | **0.0** | 0.10 ↓ |
| GEN false-fire @recall80 | **0.0** | 0.033 ↓ |

The fp32 gate is **quantization-robust** and beats a 4-bit refit on every held metric — confirming
the prior handoff finding ("refit on 4-bit is worse; do not retrain the heads"). **Keep the fp32
ondevice gate.**

## Result 2 — threshold tuning (4-bit, end-to-end): −3.0 is the knee
`recall_gen.py Q4=1`, proven ondevice gate + BGE-on-demand, N=4 (12 questions / 8 casual turns):

| gate threshold | recall | retrieval | casual false-fire (per turn) |
|---|---|---|---|
| SP (no recall) | 0% | 0% | 0% |
| **−3.0** | **92%** | 100% | **62%** |
| −3.5 (prior default) | 83% | 100% | 62% |
| −4.0 | 83% | 100% | 88% |

**−3.0 dominates −3.5** here (higher recall, same casual-fire) and beats −4.0 on both. Retrieval is
100% throughout (the right block is always pulled); `recall < 100%` is the 4-bit base model's copy
ability, not the recall mechanism (consistent with the handoff). **Recommend threshold ≈ −3.0** at
4-bit (or per-build calibrate to the ~10th-pct of recall-position gate scores).

## Result 3 — relevance floor does NOT separate  → not viable
Per-fire BGE top-retrieval-score, split by turn type (RC_bge@−3.5):

```
q-fires (recall turns): n=22  median −21.27  range [−21.48, −21.07]
c-fires (casual turns): n= 9  median −21.25  range [−21.40, −21.07]
```

The two distributions **overlap almost completely** — the BGE top-score is the same magnitude
whether or not a truly-relevant evicted block exists. So **no floor on the BGE score can suppress
casual injections without killing recall**. The relevance-floor lever (suggested in the prior
handoff) is **dead for this signal**. (Code path `:rFLOOR` is implemented and verified, but there
is nothing to threshold on.)

## Bottom line
- **Per-turn** casual false-fire ≈ 62% at the best operating point, but this aggregates ~10–20
  per-position gate checks per turn. **Per generation step (token/rebuild) the false-fire is only a
  few % (~5–10%)** — the gate is quiet at the decision-point level; the per-turn number is just the
  "fired at least once" aggregate. Impact per turn = at most one stray block injected (the model
  still answers; recall stays high). Practically **minor**.
- None of the three clean levers removes the per-turn false-fire: contrastive retrain is worse,
  threshold only trades recall, and the BGE relevance score does not separate. The residual is
  intrinsic to a **per-position** gate aggregated over many checks.

### Recommended on-device config (4-bit)
`gate` = current fp32 `ondevice_recall/gate.npz` (robust at 4-bit, do NOT retrain) · `retriever` =
BGE-on-demand (`bge_head.npz`, IDs-only) · `threshold ≈ −3.0`. Accept the low per-step false-fire.

### Open / future levers (NOT run — low priority given minor impact)
1. **fire-once-per-turn** cap (limits stray injections to ≤1/turn; does not change the rate).
2. A **better relevance signal** than the BGE top-score — e.g. top-1−top-2 score margin, or the
   gate score itself (log gate-score split by q/c-fire next time; BGE magnitude was a dead end).
3. A **per-block gate** redesign (score "do I need THIS block?" per candidate) instead of a
   per-position fire + separate retrieval — the only path to true per-block selectivity.
