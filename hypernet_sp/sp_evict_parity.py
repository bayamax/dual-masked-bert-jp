"""DECISIVE TEST + parity reference for the MLX-Swift SP-evict degeneration report
(GENONCE_DEGENERATION_REPORT.md): run the PyTorch reference _gen_once SP-evict path with
identical weights, GREEDY sampling, and NO repetition/policy guards, on the reporter's
exact trivial prompt — then dump stage-by-stage artifacts so the Swift port can be diffed
mechanically:

  parity_reference.npz
    q_ids            prompt token ids (incl. chat wrapper + <think>)
    sp0              the EMPTY-PAST soft prompt (pooler on 0 kept tokens) — fp32 [32,1536]
    sp0_norms        per-vector L2 norms (must equal |out_scale|)
    block0_checksum  sum of the first injected block embedding (sp + nothing)
    greedy_tokens    first 64 greedy token ids after the prompt
  plus a printed human-readable summary.

A Swift port that diverges should compare IN THIS ORDER: sp0 (NaN check first — softmax
over an EMPTY key set is the leading suspect for trivial prompts where kept==0), sp0_norms,
block checksum, then token-by-token.

Run next to fft_hf/ and fft_out/pooler.pt:  python3 sp_evict_parity.py
"""
import json, os, sys
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from attn_export3_torch import load_pooler

PROMPT = "What is 47 multiplied by 9?"
STEPS = 200
RW, C = 512, 64


@torch.no_grad()
def main():
    torch.set_num_threads(os.cpu_count())
    tok = AutoTokenizer.from_pretrained("fft_hf")
    llm = AutoModelForCausalLM.from_pretrained("fft_hf", dtype=torch.float32).eval()
    embT = llm.get_input_embeddings()
    pooler = load_pooler()

    def emb(ids):
        return embT(torch.tensor([ids])) if ids else torch.zeros(1, 0, pooler.H)

    q_ids = tok.encode(f"<｜User｜>{PROMPT}<｜Assistant｜>", add_special_tokens=True) \
        + tok.encode("<think>\n", add_special_tokens=False)

    # --- stage 1: the empty-past soft prompt (kept == 0 on a first turn) ---------------
    sp0 = pooler(torch.zeros(1, 0, pooler.H))
    nan = bool(torch.isnan(sp0).any())
    norms = sp0[0].norm(dim=-1)
    print(f"[stage1] empty-past SP: shape={tuple(sp0.shape)}  NaN={nan}  "
          f"norms min/max={float(norms.min()):.6f}/{float(norms.max()):.6f}  "
          f"(|out_scale|={float(pooler.out_scale.abs()):.6f})")

    # --- stage 2: SP-evict generation, greedy, NO guards -------------------------------
    cache = DynamicCache()
    llm(input_ids=torch.tensor([q_ids]), past_key_values=cache, use_cache=True)
    MQ = cache.get_seq_length()
    gen, kept, absorbed = [], [], 0
    toks = []
    done = False
    while len(gen) < STEPS and not done:
        c0 = len(gen); R = min(c0, RW); nd_end = c0 - R
        if nd_end > absorbed:
            kept.extend(gen[absorbed:nd_end]); absorbed = nd_end
        sp = pooler(emb(kept).float())
        parts = [sp] + ([emb(gen[c0 - R:c0])] if R > 0 else [])
        block = torch.cat(parts, 1)
        if c0 == 0:
            block0_checksum = float(block.sum())
        cache.crop(MQ)
        last = llm(inputs_embeds=block, past_key_values=cache, use_cache=True).logits[:, -1, :]
        for _ in range(min(C, STEPS - len(gen))):
            t = int(last[0].argmax())                  # GREEDY — deterministic, no guards
            if t == tok.eos_token_id:
                done = True; break
            gen.append(t); toks.append(t)
            last = llm(inputs_embeds=emb([t]), past_key_values=cache,
                       use_cache=True).logits[:, -1, :]
        if done:
            break
    text = tok.decode(gen)
    print(f"\n[stage2] greedy SP-evict output ({len(gen)} tok):\n{text}")
    words = sum(1 for w in text.split() if any(c.isalpha() for c in w))
    coherent = words >= len(text.split()) * 0.7 and len(text.split()) > 20
    verdict = []
    verdict.append("coherent English reasoning (NOT punctuation noise)" if coherent
                   else "DEGENERATE — investigate the recipe")
    if "423" in text:
        verdict.append("and reaches the correct product 423")
    print(f"\nVERDICT: torch reference SP-evict output is {'; '.join(verdict)}.")
    if coherent:
        print("=> RECIPE/WEIGHTS ARE CLEAN. The ,1!#!!!! artifact is Swift-port fidelity.")

    np.savez("parity_reference.npz",
             prompt=np.array(PROMPT), q_ids=np.array(q_ids),
             sp0=sp0[0].numpy().astype(np.float32),
             sp0_norms=norms.numpy().astype(np.float32),
             block0_checksum=np.array([block0_checksum]),
             greedy_tokens=np.array(toks))
    print(f"\nsaved parity_reference.npz "
          f"(q_ids={len(q_ids)}, sp0=32x{pooler.H}, greedy_tokens={len(toks)})")
    print("SP_EVICT_PARITY_DONE")


if __name__ == "__main__":
    main()
