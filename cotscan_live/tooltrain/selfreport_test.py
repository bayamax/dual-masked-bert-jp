"""selfreport_test.py — does the full-KV model itself reliably say 'I don't know' on gaps?
If yes, its own output is the label (gap=expresses uncertainty, no-call=confident answer).
Test on a TRUTH-labeled diverse set; measure uncertainty-rate per true class."""
import os, re, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
HERE=os.path.dirname(os.path.abspath(__file__)); FFT=os.path.join(HERE,"..","faithful_run","fft_hf")
tok=AutoTokenizer.from_pretrained(FFT)
try: m=AutoModelForCausalLM.from_pretrained(FFT,torch_dtype=torch.float32)
except TypeError: m=AutoModelForCausalLM.from_pretrained(FFT,dtype=torch.float32)
m=m.eval(); bos=tok.bos_token_id; eos=tok.eos_token_id
UNC=re.compile(r"(i don'?t know|i do not know|i'?m not sure|i am not sure|i do not have|i don'?t have (access|enough|specific|that|the)|cannot (find|provide|verify|answer|confirm)|can'?t (find|provide|verify)|not aware|no (information|record|data|reliable)|as of my (last|knowledge|training)|i'?m unable|i am unable|unable to (find|provide)|don'?t have access|fictional|made[- ]up|doesn'?t (exist|appear)|there is no|i couldn'?t find|not enough information|i'?m not familiar|not familiar with)",re.I)
SETS={
 "known":["What is the capital of France?","Who wrote Hamlet?","What is 2 plus 2?","Which planet is the Red Planet?","What is the chemical symbol for gold?","Who painted the Mona Lisa?"],
 "fabricated":["In what year was the Treaty of Velmaran signed?","What is the capital of Brindolia?","Who won the Galvin Prize in 2019?","How tall is Mount Drennakar?","Who is the CEO of the Brammelton Corporation?","What language is spoken in Ulventhia?"],
 "obscure_real":["What is the population of Tallinn?","What is the elevation of Denver?","Who discovered the planet Neptune?","What is the population of Reykjavik?","In what year was the Tokyo Skytree completed?","Who won the Eurovision Song Contest in 2014?"],
 "creative":["Write a short poem about the ocean.","Brainstorm three gift ideas for a teacher.","Write a one-line slogan for a coffee shop."],
 "math":["What is 13 times 4?","What is 144 divided by 12?","If I have 5 apples and eat 2, how many remain?"],
 "chitchat":["I burnt my toast this morning.","I'm thinking of repainting my room.","I had a great weekend."],
}
LOG=[]
def log(x): print(x,flush=True); LOG.append(x); open(os.path.join(HERE,"selfreport.log"),"w").write("\n".join(LOG))
@torch.no_grad()
def ans(q,n=90):
    ids=[bos]+tok.encode(f"<｜User｜>{q}<｜Assistant｜>",add_special_tokens=False)
    out=m.generate(input_ids=torch.tensor([ids]),max_new_tokens=n,do_sample=False,pad_token_id=eos)
    return tok.decode(out[0,len(ids):],skip_special_tokens=True)
for cls,qs in SETS.items():
    u=0
    for q in qs:
        a=ans(q); unc=bool(UNC.search(a)); u+=unc
        log(f"[{cls:12} unc={int(unc)}] {q[:38]} -> {a[:90].strip()}")
    log(f"  == {cls}: uncertain {u}/{len(qs)} ==\n")
log("[DONE] (gap classes=fabricated/obscure_real want HIGH uncertain; others want LOW)")
