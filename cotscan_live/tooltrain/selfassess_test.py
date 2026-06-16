"""selfassess_test.py — ask the model to self-judge CONFIDENT vs LOOKUP. Cleaner label than regex.
"""
import os, re, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
HERE=os.path.dirname(os.path.abspath(__file__)); FFT=os.path.join(HERE,"..","faithful_run","fft_hf")
tok=AutoTokenizer.from_pretrained(FFT)
try: m=AutoModelForCausalLM.from_pretrained(FFT,torch_dtype=torch.float32)
except TypeError: m=AutoModelForCausalLM.from_pretrained(FFT,dtype=torch.float32)
m=m.eval(); bos=tok.bos_token_id; eos=tok.eos_token_id
SETS={
 "known":["What is the capital of France?","Who wrote Hamlet?","What is 2 plus 2?","Which planet is the Red Planet?","What is the chemical symbol for gold?","Who painted the Mona Lisa?"],
 "fabricated":["In what year was the Treaty of Velmaran signed?","What is the capital of Brindolia?","Who won the Galvin Prize in 2019?","How tall is Mount Drennakar?","Who is the CEO of the Brammelton Corporation?","What language is spoken in Ulventhia?"],
 "obscure_real":["What is the population of Tallinn?","What is the elevation of Denver?","Who discovered the planet Neptune?","What is the population of Reykjavik?","In what year was the Tokyo Skytree completed?","Who won the Eurovision Song Contest in 2014?"],
 "creative":["Write a short poem about the ocean.","Brainstorm three gift ideas for a teacher.","Write a one-line slogan for a coffee shop."],
 "math":["What is 13 times 4?","What is 144 divided by 12?","If I have 5 apples and eat 2, how many remain?"],
 "chitchat":["I burnt my toast this morning.","I'm thinking of repainting my room.","I had a great weekend."],
}
INSTR=("Decide whether you can answer the user's request confidently from your own knowledge, "
       "or whether it needs an external/up-to-date fact you should look up. "
       "Reply with ONLY one word: CONFIDENT or LOOKUP.")
LOG=[]
def log(x): print(x,flush=True); LOG.append(x); open(os.path.join(HERE,"selfassess.log"),"w").write("\n".join(LOG))
@torch.no_grad()
def judge(q,n=12):
    p=f"<｜User｜>{INSTR}\n\nUser request: \"{q}\"<｜Assistant｜>"
    ids=[bos]+tok.encode(p,add_special_tokens=False)+tok.encode("<think>\n\n</think>\n\n",add_special_tokens=False)
    out=m.generate(input_ids=torch.tensor([ids]),max_new_tokens=n,do_sample=False,pad_token_id=eos)
    t=tok.decode(out[0,len(ids):],skip_special_tokens=True).upper()
    return "LOOKUP" if ("LOOKUP" in t and "CONFIDENT" not in t.split("LOOKUP")[0]) else ("LOOKUP" if "LOOKUP" in t and "CONFIDENT" not in t else "CONFIDENT"), t.strip()[:30]
for cls,qs in SETS.items():
    lk=0
    for q in qs:
        lab,raw=judge(q); lk+= (lab=="LOOKUP")
        log(f"[{cls:12} {lab:9}] {q[:36]} -> raw:{raw}")
    log(f"  == {cls}: LOOKUP {lk}/{len(qs)} ==\n")
log("[DONE] gap(fabricated/obscure) want LOOKUP high; others want CONFIDENT")
