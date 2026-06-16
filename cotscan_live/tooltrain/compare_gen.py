"""compare_gen.py — did the tool-call LoRA BREAK general generation? Compare BASE (adapter
disabled) vs SFT (adapter enabled) on non-tool tasks, same prompts, greedy. CPU."""
import os, torch, time
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
HERE=os.path.dirname(os.path.abspath(__file__)); FFT=os.path.join(HERE,"..","faithful_run","fft_hf"); ADP="/tmp/lora"
tok=AutoTokenizer.from_pretrained(FFT)
try: base=AutoModelForCausalLM.from_pretrained(FFT,torch_dtype=torch.float32)
except TypeError: base=AutoModelForCausalLM.from_pretrained(FFT,dtype=torch.float32)
m=PeftModel.from_pretrained(base,ADP).eval(); bos=tok.bos_token_id; eos=tok.eos_token_id
LOG=[]
def log(x): print(x,flush=True); LOG.append(x); open(os.path.join(HERE,"compare_gen.log"),"w").write("\n".join(LOG))
@torch.no_grad()
def g(q):
    ids=[bos]+tok.encode(f"<｜User｜>{q}<｜Assistant｜>",add_special_tokens=False)+tok.encode("<think>\n\n</think>\n\n",add_special_tokens=False)
    out=m.generate(input_ids=torch.tensor([ids]),max_new_tokens=70,do_sample=False,pad_token_id=eos)
    return tok.decode(out[0,len(ids):],skip_special_tokens=True).strip()
BATTERY=["Who wrote Hamlet?","What is 2 plus 2?","What is the capital of France?","Name three primary colors.",
 "Write a two-line poem about autumn.","Explain photosynthesis in one sentence.","How are you today?",
 "I just got back from a nice walk.","What is the largest planet in our solar system?"]
for q in BATTERY:
    with m.disable_adapter():
        b=g(q)
    a=g(q)
    log(f"\nQ: {q}\n  BASE : {b[:180]}\n  SFT  : {a[:180]}")
log("\n[DONE]")
