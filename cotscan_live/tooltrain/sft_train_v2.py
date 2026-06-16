"""sft_train_v2.py — milder tool-call SFT to PRESERVE the original generation.
Anti-forgetting: small LoRA (r8/a16), low LR (5e-5), 2 epochs, + REPLAY (base model's OWN answers
on general prompts, self-distilled) mixed in to anchor the original distribution.
Evals search-timing AND base-vs-SFT general gen to confirm reduced drift.
"""
import os, json, time, random, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
DEV="cuda" if torch.cuda.is_available() else "cpu"; HERE=os.path.dirname(os.path.abspath(__file__))
def log(*a): print(*a,flush=True)
REPLAY_PROMPTS=["What is 2 plus 2?","Who wrote Hamlet?","Name three primary colors.","What is the largest planet?",
 "I just got back from a nice walk.","How are you today?","Tell me a fun fact.","What's your favorite season?",
 "Explain gravity simply.","What is 15 minus 7?","Give me a tip for staying focused.","I'm feeling tired today.",
 "What's the capital of Spain?","How do I make tea?","Recommend a hobby.","What is water made of?",
 "I had pasta for dinner.","What day comes after Monday?","Count from 1 to 5.","Say hello.",
 "What's the opposite of hot?","Name a fruit.","I love rainy days.","What is 9 times 3?",
 "Describe a cat in one sentence.","What's a good morning routine?","Thanks for your help!","What's 100 divided by 4?",
 "I'm planning a birthday party.","What color is the sky?","Tell me a short joke.","What is the freezing point of water?"]
HELD_GAP=["Who won the FIFA World Cup in 1998?","What is the population of Reykjavik?","When was the Brooklyn Bridge opened?",
 "Who painted Guernica and in what year?","Which company manufactured the first mass-produced car?"]
HELD_KNOWN=["What is the capital of France?","What is 2 plus 2?","Who wrote Hamlet?","Which planet is closest to the sun?"]
HELD_CHIT=["I had a really great weekend.","How are you doing today?","I just made some coffee."]
def main():
    t0=time.time(); tok=AutoTokenizer.from_pretrained("fft_hf"); bos=tok.bos_token_id; eos=tok.eos_token_id
    dt=torch.bfloat16 if (DEV=="cuda" and torch.cuda.is_bf16_supported()) else torch.float32
    try: m=AutoModelForCausalLM.from_pretrained("fft_hf",torch_dtype=dt)
    except TypeError: m=AutoModelForCausalLM.from_pretrained("fft_hf",dtype=dt)
    m=m.to(DEV)
    # 1) REPLAY: base answers (no LoRA yet) -> anchor original behavior
    @torch.no_grad()
    def baseans(q,n=64):
        ids=[bos]+tok.encode(f"<｜User｜>{q}<｜Assistant｜>",add_special_tokens=False)+tok.encode("<think>\n\n</think>\n\n",add_special_tokens=False)
        out=m.generate(input_ids=torch.tensor([ids],device=DEV),max_new_tokens=n,do_sample=False,pad_token_id=eos)
        return tok.decode(out[0,len(ids):],skip_special_tokens=True).strip()
    m.eval(); replay=[]
    for q in REPLAY_PROMPTS:
        a=baseans(q); replay.append({"prompt":f"<｜User｜>{q}<｜Assistant｜>","completion":f"<think>I can answer directly.</think>{a}","kind":"replay"})
    log(f"[replay] {len(replay)} base self-distilled rows")
    # 2) LoRA (small) + train
    m.config.use_cache=False
    lora=LoraConfig(r=8,lora_alpha=16,lora_dropout=0.05,bias="none",task_type="CAUSAL_LM",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])
    m=get_peft_model(m,lora); m.print_trainable_parameters()
    rows=[json.loads(l) for l in open("call_sft_v2.jsonl") if l.strip()]+replay
    log(f"[data] {len(rows)} rows (tool {len(rows)-len(replay)} + replay {len(replay)})")
    def enc(r):
        p=[bos]+tok.encode(r["prompt"],add_special_tokens=False); c=tok.encode(r["completion"],add_special_tokens=False)+[eos]
        return (p+c)[:1024], ([-100]*len(p)+c)[:1024]
    data=[enc(r) for r in rows]
    opt=torch.optim.AdamW([q for q in m.parameters() if q.requires_grad],lr=5e-5)
    m.train()
    for ep in range(2):
        random.Random(ep).shuffle(data); tot=0.0
        for ids,lab in data:
            out=m(input_ids=torch.tensor([ids],device=DEV),labels=torch.tensor([lab],device=DEV)); out.loss.backward()
            torch.nn.utils.clip_grad_norm_([q for q in m.parameters() if q.requires_grad],1.0); opt.step(); opt.zero_grad(); tot+=out.loss.item()
        log(f"[ep {ep+1}/2] loss={tot/len(data):.4f} t={time.time()-t0:.0f}s")
    m.save_pretrained(os.path.join(HERE,"lora_adapter_v2")); log("[saved v2 adapter]")
    # 3) eval: search-timing + base-vs-SFT gen
    m.eval(); m.config.use_cache=True
    @torch.no_grad()
    def gen(q,n=70,adapter=True):
        ids=[bos]+tok.encode(f"<｜User｜>{q}<｜Assistant｜>",add_special_tokens=False)
        cm=(lambda: m.disable_adapter()) if not adapter else None
        if adapter: out=m.generate(input_ids=torch.tensor([ids],device=DEV),max_new_tokens=n,do_sample=False,pad_token_id=eos)
        else:
            with m.disable_adapter(): out=m.generate(input_ids=torch.tensor([ids],device=DEV),max_new_tokens=n,do_sample=False,pad_token_id=eos)
        return tok.decode(out[0,len(ids):],skip_special_tokens=False)
    def fired(t): return "<search>" in t
    gf=[fired(gen(q)) for q in HELD_GAP]; kf=[fired(gen(q)) for q in HELD_KNOWN]; cf=[fired(gen(q)) for q in HELD_CHIT]
    log(f"\n=== TIMING === gap_fire={sum(gf)}/{len(gf)} known_fire={sum(kf)}/{len(kf)} chit_fire={sum(cf)}/{len(cf)}")
    log("\n=== GEN base vs SFTv2 ===")
    for q in ["What is 2 plus 2?","I just got back from a nice walk.","Who wrote Hamlet?","What is the capital of France?","Recommend a hobby."]:
        b=gen(q,adapter=False); a=gen(q,adapter=True)
        import re
        clean=lambda s: re.sub(r"<[^>]*>","",s).strip()[:150]
        log(f"\nQ: {q}\n  BASE: {clean(b)}\n  SFT2: {clean(a)}")
    json.dump({"gap_fire":f"{sum(gf)}/{len(gf)}","known_fire":f"{sum(kf)}/{len(kf)}","chit_fire":f"{sum(cf)}/{len(cf)}"},
              open(os.path.join(HERE,"sft_v2_eval.json"),"w"),indent=1)
    open(os.path.join(HERE,".done"),"w").write("ok"); log(f"DONE t={time.time()-t0:.0f}s")
if __name__=="__main__": main()
