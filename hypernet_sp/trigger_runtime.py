"""Runtime wrapper for the attention-mass retrieval trigger (attn_trigger3.joblib).

`TriggerScorer.score(sp, turn_ids)` returns P(this turn needs a value that now lives only
in the SP). The runtime fires external retrieval (L1/L2/L3 re-injection into the raw
window) when the score clears a threshold — replacing the heuristic "is this referential?"
intent guesswork with the model's own SP-attention signal (STATUS §4).

Cost: one extra prompt-length forward with eager attention (no generation, no lm_head use).
On Metal/MLX this is milliseconds; the MLX port only needs QK for the 6 selected
(layer, head) pairs, all of which sit at layer <= 22 of 28.

__main__ = threshold characterisation: recomputes the 60 probe examples' scores with the
SAVED trigger and prints the operating table (precision / recall / FPR per threshold) so
the app can pick the right cost trade-off. A missed fire = a confabulated number (bad);
a false fire = one wasted retrieval (cheap) -> recommend the low threshold.
Run next to fft_hf/, fft_out/pooler.pt, attn_probe3.npz, attn_trigger3.joblib.
"""
import numpy as np


class TriggerScorer:
    def __init__(self, llm, bos_id, bundle_path="attn_trigger3.joblib"):
        import joblib, torch
        self.llm, self.bos, self.torch = llm, bos_id, torch
        b = joblib.load(bundle_path)
        self.heads, self.clf, self.mu, self.sd = b["heads"], b["clf"], b["mu"], b["sd"]
        self.max_layer = max(l for l, _ in self.heads)
        self.embT = llm.get_input_embeddings()

    def _masses(self, sp, t2ids):
        torch = self.torch
        n_sp = sp.shape[0]; off = 1 + n_sp
        inp = torch.cat([self.embT(torch.tensor([self.bos])),
                         torch.tensor(np.asarray(sp), dtype=self.embT.weight.dtype),
                         self.embT(torch.tensor(t2ids, dtype=torch.long))], 0)[None]
        with torch.no_grad():
            out = self.llm(inputs_embeds=inp, output_attentions=True, use_cache=False)
        f = []
        for l, h in self.heads:
            A = out.attentions[l][0, h]                       # (Lq, Lk)
            f.append(float(A[off:, 1:1 + n_sp].sum(-1).mean()))
        return np.array(f)

    def score(self, sp, t2ids):
        x = (self._masses(sp, t2ids) - self.mu) / self.sd
        return float(self.clf.predict_proba([x])[0, 1])


def main():
    import torch
    from transformers import AutoModelForCausalLM
    torch.set_num_threads(__import__("os").cpu_count())
    d = np.load("attn_probe3.npz", allow_pickle=True)
    n, bos = int(d["n"][0]), int(d["bos"][0])
    cat = [str(c) for c in d["cat"]]
    llm = AutoModelForCausalLM.from_pretrained("fft_hf", dtype=torch.float32,
                                               attn_implementation="eager").eval()
    sc = TriggerScorer(llm, bos)
    import time
    rows = []
    for i in range(n):
        t0 = time.time()
        pr = sc.score(d[f"sp_{i}"], d[f"ref_{i}"])
        pc = sc.score(d[f"sp_{i}"], d[f"ctrl_{i}"])
        rows.append((pr, pc))
        if i == 0:
            print(f"per-call walltime (CPU fp32, whole-model eager): {time.time() - t0:.1f}s "
                  f"(Metal: ms-scale; QK-only port needs layers <= {sc.max_layer})")
    pr = np.array([r[0] for r in rows]); pc = np.array([r[1] for r in rows])
    print(f"\nscores: ref mean={pr.mean():.3f} ctrl mean={pc.mean():.3f} | pairwise ref>ctrl "
          f"{int((pr > pc).sum())}/{n}")
    print("\n== operating table (fire retrieval when score >= th) ==")
    print("  th    recall(ref fired)  FPR(ctrl fired)  precision")
    for th in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        tp = int((pr >= th).sum()); fp = int((pc >= th).sum())
        prec = tp / max(tp + fp, 1)
        print(f"  {th:.1f}   {tp:>2}/{n} = {tp / n:.2f}        {fp:>2}/{n} = {fp / n:.2f}       {prec:.2f}")
    miss = [(cat[i], float(pr[i])) for i in range(n) if pr[i] < 0.4]
    print(f"\nreferential turns scoring < 0.4 (would be missed at the recommended th): {miss or 'none'}")
    print("TRIGGER_RUNTIME_DONE")


if __name__ == "__main__":
    main()
