"""eval_quality.py — base vs v3 general-generation quality (did tool-call SFT degrade it?).
Diverse battery; base=adapter disabled (pre-closed think), v3=adapter (natural). Strip tags, compare."""
import os, re, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
HERE=os.path.dirname(os.path.abspath(__file__)); FFT=os.path.join(HERE,"..","faithful_run","fft_hf"); ADP=os.environ.get("ADP","/tmp/lora_v3")
tok=AutoTokenizer.from_pretrained(FFT)
try: base=AutoModelForCausalLM.from_pretrained(FFT,torch_dtype=torch.float32)
except TypeError: base=AutoModelForCausalLM.from_pretrained(FFT,dtype=torch.float32)
m=PeftModel.from_pretrained(base,ADP).eval(); bos=tok.bos_token_id; eos=tok.eos_token_id
LOG=[]
def log(x): print(x,flush=True); LOG.append(x); open(os.path.join(HERE,"eval_quality.log"),"w").write("\n".join(LOG))
clean=lambda s: re.sub(r"<[^>]*>","",s).strip()
@torch.no_grad()
def gbase(q,n=70):
    ids=[bos]+tok.encode(f"<｜User｜>{q}<｜Assistant｜>",add_special_tokens=False)+tok.encode("<think>\n\n</think>\n\n",add_special_tokens=False)
    with m.disable_adapter(): out=m.generate(input_ids=torch.tensor([ids]),max_new_tokens=n,do_sample=False,pad_token_id=eos)
    return clean(tok.decode(out[0,len(ids):],skip_special_tokens=True))
@torch.no_grad()
def gv3(q,n=70):
    ids=[bos]+tok.encode(f"<｜User｜>{q}<｜Assistant｜>",add_special_tokens=False)
    out=m.generate(input_ids=torch.tensor([ids]),max_new_tokens=n,do_sample=False,pad_token_id=eos)
    return clean(tok.decode(out[0,len(ids):],skip_special_tokens=True))
BATTERY=[("chitchat","I burnt my toast this morning."),("chitchat","I'm thinking of repainting my room."),
 ("known","What is the capital of Argentina?"),("known","Who wrote The Great Gatsby?"),
 ("math","What is 13 times 4?"),("math","What is 144 divided by 12?"),
 ("instruct","List three benefits of regular exercise."),("instruct","Give me one tip for better sleep."),
 ("explain","Explain how rainbows form in one sentence."),("explain","What is inflation?"),
 ("creative","Write a short two-line poem about the ocean.")]
for kind,q in BATTERY:
    b=gbase(q); v=gv3(q)
    log(f"\n[{kind}] {q}\n  BASE: {b[:170]}\n  V3  : {v[:170]}")
log("\n[DONE]")
