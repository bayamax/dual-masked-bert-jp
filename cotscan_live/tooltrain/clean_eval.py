"""clean_eval.py — HELD-OUT (disjoint from training) eval of search-timing. Questions here are
NOT in any training bank, to test generalization (avoid memorization/local-optimum false signal).
Usage: ADP=/path/to/adapter python clean_eval.py"""
import os, torch, re
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
HERE=os.path.dirname(os.path.abspath(__file__)); FFT=os.path.join(HERE,"..","faithful_run","fft_hf"); ADP=os.environ.get("ADP","/tmp/lora_v3")
# DISJOINT held-out (none of these appear in training banks)
GAP=["When was the Golden Gate Bridge completed?","What is the population of Tallinn?",
 "Who won the Eurovision Song Contest in 2014?","Who is the current president of Iceland?",
 "When did the Great Fire of Rome occur?","What is the elevation of Denver, Colorado?",
 "Who discovered the planet Neptune?","In what year was the Tokyo Skytree completed?"]
KNOWN=["What is the capital of Argentina?","Who wrote The Great Gatsby?","What is the chemical symbol for silver?",
 "What is the capital of Poland?","Who proposed the theory of evolution?","How many legs does a spider have?",
 "What is the capital of Egypt?","What is the largest mammal?"]
CHIT=["I burnt my toast this morning.","My favorite team won last night.","I'm thinking of repainting my room.",
 "I can't stop listening to this song.","My plants are finally blooming."]
def main():
    tok=AutoTokenizer.from_pretrained(FFT)
    try: base=AutoModelForCausalLM.from_pretrained(FFT,torch_dtype=torch.float32)
    except TypeError: base=AutoModelForCausalLM.from_pretrained(FFT,dtype=torch.float32)
    m=PeftModel.from_pretrained(base,ADP).eval(); bos=tok.bos_token_id; eos=tok.eos_token_id
    LOG=[]
    def log(x): print(x,flush=True); LOG.append(x); open(os.path.join(HERE,"clean_eval.log"),"w").write("\n".join(LOG))
    @torch.no_grad()
    def fires(q):
        ids=[bos]+tok.encode(f"<｜User｜>{q}<｜Assistant｜>",add_special_tokens=False)
        out=m.generate(input_ids=torch.tensor([ids]),max_new_tokens=48,do_sample=False,pad_token_id=eos)
        t=tok.decode(out[0,len(ids):],skip_special_tokens=False); return "<search>" in t, re.sub(r"<[^>]*>","",t).strip()[:80]
    res={}
    for name,qs,want in [("GAP(held)",GAP,"high"),("KNOWN(held)",KNOWN,"0"),("CHIT(held)",CHIT,"0")]:
        fl=[]
        for q in qs:
            f,t=fires(q); fl.append(f); log(f"[{name} fire={int(f)}] {q[:42]} -> {t[:60]}")
        res[name]=f"{sum(fl)}/{len(fl)}"
    log(f"\n=== HELD-OUT TIMING (want gap high, known/chit 0) ===")
    for k,v in res.items(): log(f"  {k}: {v}  (want {'high' if 'GAP' in k else '0'})")
    import json; json.dump(res,open(os.path.join(HERE,"clean_eval.json"),"w"),indent=1)
if __name__=="__main__": main()
