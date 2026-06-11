"""Greeting-degeneration isolation experiment (GENERATION_CHITCHAT_REPORT.md).

A bare greeting yields invented tasks ("Hello" -> "12." / stats lecture). Which layer?

  arm A  canon SP pipeline (AppSession)         — run separately (greeting_repro)
  arm B  fft_hf, full-KV, forced <think>        — isolates: does plain full attention
                                                   with think-forcing already invent?
  arm C  fft_hf, full-KV, NO think forcing      — isolates the forced-think factor
  arm D  ORIGINAL DeepSeek-R1-Distill-Qwen-1.5B, full-KV, forced <think>
                                                 — isolates FFT chat-erosion (the FFT
                                                   corpus was math-CoT only)
  arm E  original weights, NO think forcing     — base behaviour reference

3 samples per arm at temp 0.6 (the report's regime) so we measure a RATE, not a coin flip.
Run next to fft_hf/:  python3 greeting_isolation.py
"""
import os, sys
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPT = "Hello"
N, CAP, TEMP = 3, 400, 0.6
GREET = ("hello", "hi ", "hi!", "hey", "help you", "assist", "how can i", "nice to",
         "good morning", "good day", "welcome", "what can i do", "how are you")


@torch.no_grad()
def gen_full_kv(llm, tok, force_think, seed):
    torch.manual_seed(seed)
    text = f"<｜User｜>{PROMPT}<｜Assistant｜>" + ("<think>\n" if force_think else "")
    ids = tok.encode(text, add_special_tokens=True)
    out = list(ids)
    past = None
    inp = torch.tensor([out])
    for _ in range(CAP):
        o = llm(input_ids=inp, past_key_values=past, use_cache=True)
        past = o.past_key_values
        t = int(torch.multinomial(F.softmax(o.logits[0, -1] / TEMP, -1), 1))
        if t == tok.eos_token_id:
            break
        out.append(t)
        inp = torch.tensor([[t]])
    return tok.decode(out[len(ids):])


def judge(body):
    """greeting-appropriate? answer (post-think if present) contains conversational
    greeting markers and is not a bare number / lecture."""
    ans = body.split("</think>")[-1] if "</think>" in body else body
    a = ans.strip().lower()
    if not a:
        return False, "(empty)"
    bare_number = len(a) < 12 and any(c.isdigit() for c in a)
    greet = any(g in a for g in GREET)
    return (greet and not bare_number), ans.strip()[:110]


def run_arm(name, llm, tok, force_think):
    ok = 0
    for i in range(N):
        body = gen_full_kv(llm, tok, force_think, seed=80 + i)
        good, snippet = judge(body)
        ok += good
        print(f"  [{name} s{i}] {'GOOD' if good else 'BAD '} | {snippet!r}", flush=True)
    print(f"  {name}: {ok}/{N} greeting-appropriate\n", flush=True)
    return ok


def main():
    torch.set_num_threads(os.cpu_count())
    tok = AutoTokenizer.from_pretrained("fft_hf")
    fft = AutoModelForCausalLM.from_pretrained("fft_hf", dtype=torch.float32).eval()
    print("== arm B: FFT weights, full-KV, forced <think> ==", flush=True)
    b = run_arm("B", fft, tok, True)
    print("== arm C: FFT weights, full-KV, no think forcing ==", flush=True)
    c = run_arm("C", fft, tok, False)
    del fft

    base_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    print(f"== loading original weights ({base_id}) ==", flush=True)
    base = AutoModelForCausalLM.from_pretrained(base_id, dtype=torch.float32).eval()
    btok = AutoTokenizer.from_pretrained(base_id)
    print("== arm D: ORIGINAL weights, full-KV, forced <think> ==", flush=True)
    d = run_arm("D", base, btok, True)
    print("== arm E: ORIGINAL weights, full-KV, no think forcing ==", flush=True)
    e = run_arm("E", base, btok, False)

    print("=" * 60)
    print(f"B(fft+think)={b}/{N}  C(fft)={c}/{N}  D(orig+think)={d}/{N}  E(orig)={e}/{N}")
    print("interpretation: D>>B => FFT eroded chat | B~D both low => forced-think/base | "
          "C,E high => think forcing is the trigger")
    print("GREETING_ISOLATION_DONE")


if __name__ == "__main__":
    main()
