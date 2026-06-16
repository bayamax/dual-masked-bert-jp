"""sft_train_v3.py — fix OVER-CALLING via many NEGATIVES (no-call). Big bank of famous/common
facts + chitchat the model should answer DIRECTLY (no <search>); answers self-distilled full-KV
to anchor clean generation. gap traces kept from call_sft_v2. Middle LoRA (r16, LR7e-5, 2ep).
Evals timing (esp. known non-fire) + gen.
"""
import os, json, time, random, re, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
DEV="cuda" if torch.cuda.is_available() else "cpu"; HERE=os.path.dirname(os.path.abspath(__file__))
def log(*a): print(*a,flush=True)
COUNTRIES=["France","Japan","Italy","Germany","Spain","Canada","Brazil","Egypt","Kenya","India",
 "China","Russia","Australia","Mexico","Greece","Portugal","Norway","Sweden","Thailand","Turkey"]
WORKS=[("Hamlet","Shakespeare"),("Macbeth","Shakespeare"),("Romeo and Juliet","Shakespeare"),
 ("Pride and Prejudice","Jane Austen"),("War and Peace","Tolstoy"),("1984","Orwell"),
 ("The Odyssey","Homer"),("Don Quixote","Cervantes"),("Hamlet","Shakespeare"),("Oliver Twist","Dickens")]
ART=[("the Mona Lisa","painted"),("Guernica","painted"),("the Sistine Chapel ceiling","painted"),
 ("the Ninth Symphony","composed"),("The Starry Night","painted")]
SCI=["What is the chemical symbol for gold?","What is the chemical symbol for oxygen?","What is the largest planet?",
 "Which planet is closest to the sun?","What gas do plants absorb?","What is the boiling point of water in Celsius?",
 "What is the freezing point of water in Celsius?","How many bones are in the adult human body?","What is H2O?",
 "What is the speed of light approximately?","What is the powerhouse of the cell?","What force keeps us on the ground?"]
MATH=["What is 2 plus 2?","What is 15 minus 7?","What is 9 times 3?","What is 100 divided by 4?","What is 12 times 11?",
 "What is the square root of 81?","What is 7 plus 8?","What is half of 50?"]
GEO=["What is the largest ocean on Earth?","What is the tallest mountain on Earth?","What is the longest river in the world?",
 "How many continents are there?","What is the largest desert?","What is the smallest planet?"]
DEF=["What is photosynthesis?","What is gravity?","What is a noun?","What is the capital city of a country?",
 "What does CPU stand for?","What is an ecosystem?","What is evaporation?"]
CHIT=["I just adopted two kittens.","The weather has been lovely lately.","I'm a bit stressed about a deadline.",
 "I tried a great ramen place yesterday.","Hey, how's it going?","I love hiking on weekends.","Good morning!",
 "I watched a great movie last night.","Thanks for your help!","I just got back from a nice walk.",
 "I'm feeling pretty good today.","I started learning the guitar.","My neighbor got a new puppy.",
 "I had pasta for dinner.","I'm planning a birthday party.","I love rainy days.","Just finished a long week.",
 "I made some coffee.","What a beautiful sunset.","I'm reading a fun book."]
def nocall_questions():
    qs=[]
    for c in COUNTRIES: qs.append(f"What is the capital of {c}?")
    for w,_ in WORKS: qs.append(f"Who wrote {w}?")
    for a,v in ART: qs.append(f"Who {v} {a}?")
    qs+=SCI+MATH+GEO+DEF
    return list(dict.fromkeys(qs))  # dedupe

HELD_GAP=["Who won the FIFA World Cup in 1998?","What is the population of Reykjavik?","When was the Brooklyn Bridge opened?",
 "Who painted Guernica and in what year?","Which company manufactured the first mass-produced car?","What is the GDP of Estonia?"]
HELD_KNOWN=["What is the capital of France?","What is 2 plus 2?","Who wrote Hamlet?","Which planet is closest to the sun?",
 "What is the chemical symbol for gold?","Who painted the Mona Lisa?"]
HELD_CHIT=["I had a really great weekend.","How are you doing today?","I just made some coffee."]

def main():
    t0=time.time(); tok=AutoTokenizer.from_pretrained("fft_hf"); bos=tok.bos_token_id; eos=tok.eos_token_id
    dt=torch.bfloat16 if (DEV=="cuda" and torch.cuda.is_bf16_supported()) else torch.float32
    try: m=AutoModelForCausalLM.from_pretrained("fft_hf",torch_dtype=dt)
    except TypeError: m=AutoModelForCausalLM.from_pretrained("fft_hf",dtype=dt)
    m=m.to(DEV).eval()
    @torch.no_grad()
    def baseans(q,n=60):
        ids=[bos]+tok.encode(f"<｜User｜>{q}<｜Assistant｜>",add_special_tokens=False)+tok.encode("<think>\n\n</think>\n\n",add_special_tokens=False)
        out=m.generate(input_ids=torch.tensor([ids],device=DEV),max_new_tokens=n,do_sample=False,pad_token_id=eos)
        return tok.decode(out[0,len(ids):],skip_special_tokens=True).strip()
    # NEGATIVES: famous facts (self-distilled direct answers) + chitchat
    nq=nocall_questions(); nocall=[]
    for q in nq:
        a=baseans(q); nocall.append({"prompt":f"<｜User｜>{q}<｜Assistant｜>","completion":f"<think>I know this; no search.</think>{a}","kind":"nocall_fact"})
    for u in CHIT:
        a=baseans(u); nocall.append({"prompt":f"<｜User｜>{u}<｜Assistant｜>","completion":f"<think>Casual; no search.</think>{a}","kind":"nocall_chit"})
    log(f"[negatives] facts={len(nq)} chit={len(CHIT)} total no-call={len(nocall)}")
    # POSITIVES: gap traces from call_sft_v2
    gap=[json.loads(l) for l in open("call_sft_v2.jsonl") if l.strip()]; gap=[r for r in gap if r["kind"]=="gap"]
    rows=gap+nocall; log(f"[data] gap={len(gap)} no-call={len(nocall)} total={len(rows)}")
    # LoRA middle
    m.config.use_cache=False
    lora=LoraConfig(r=16,lora_alpha=32,lora_dropout=0.05,bias="none",task_type="CAUSAL_LM",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])
    m=get_peft_model(m,lora); m.print_trainable_parameters()
    def enc(r):
        p=[bos]+tok.encode(r["prompt"],add_special_tokens=False); c=tok.encode(r["completion"],add_special_tokens=False)+[eos]
        return (p+c)[:1024], ([-100]*len(p)+c)[:1024]
    data=[enc(r) for r in rows]; opt=torch.optim.AdamW([q for q in m.parameters() if q.requires_grad],lr=7e-5)
    m.train()
    for ep in range(2):
        random.Random(ep).shuffle(data); tot=0.0
        for ids,lab in data:
            out=m(input_ids=torch.tensor([ids],device=DEV),labels=torch.tensor([lab],device=DEV)); out.loss.backward()
            torch.nn.utils.clip_grad_norm_([q for q in m.parameters() if q.requires_grad],1.0); opt.step(); opt.zero_grad(); tot+=out.loss.item()
        log(f"[ep {ep+1}/2] loss={tot/len(data):.4f} t={time.time()-t0:.0f}s")
    m.save_pretrained(os.path.join(HERE,"lora_adapter_v3")); log("[saved v3]")
    m.eval(); m.config.use_cache=True
    @torch.no_grad()
    def gen(q,n=70,adapter=True):
        ids=[bos]+tok.encode(f"<｜User｜>{q}<｜Assistant｜>",add_special_tokens=False)
        if adapter: out=m.generate(input_ids=torch.tensor([ids],device=DEV),max_new_tokens=n,do_sample=False,pad_token_id=eos)
        else:
            with m.disable_adapter(): out=m.generate(input_ids=torch.tensor([ids],device=DEV),max_new_tokens=n,do_sample=False,pad_token_id=eos)
        return tok.decode(out[0,len(ids):],skip_special_tokens=False)
    f=lambda t:"<search>" in t
    gf=[f(gen(q)) for q in HELD_GAP]; kf=[f(gen(q)) for q in HELD_KNOWN]; cf=[f(gen(q)) for q in HELD_CHIT]
    log(f"\n=== TIMING === gap_fire={sum(gf)}/{len(gf)} known_fire={sum(kf)}/{len(kf)} chit_fire={sum(cf)}/{len(cf)}")
    for i,q in enumerate(HELD_KNOWN): log(f"  known[{int(kf[i])}] {q}")
    log("\n=== GEN (SFTv3) ===")
    for q in ["What is 2 plus 2?","Who wrote Hamlet?","I just got back from a nice walk.","Recommend a hobby."]:
        log(f"Q:{q}\n  {re.sub(chr(60)+'[^>]*'+chr(62),'',gen(q)).strip()[:150]}")
    json.dump({"gap_fire":f"{sum(gf)}/{len(gf)}","known_fire":f"{sum(kf)}/{len(kf)}","chit_fire":f"{sum(cf)}/{len(cf)}"},open(os.path.join(HERE,"sft_v3_eval.json"),"w"),indent=1)
    open(os.path.join(HERE,".done"),"w").write("ok"); log(f"DONE t={time.time()-t0:.0f}s")
if __name__=="__main__": main()
