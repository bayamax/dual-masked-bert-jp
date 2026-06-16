"""recall_sft.py — recall-aware generation SFT (GPU pod). Teach the model to REASON in the
runtime's SP+recall context, using Dolphin-R1's existing CoTs as targets (no teacher generation).

Per example (problem P + Dolphin CoT C):
  seq = [BOS] + <｜User｜>P<｜Assistant｜> + C
  Evict the head (problem + early CoT) -> SP (pooler) + the problem block re-injected (recall).
  Train (teacher-forced) the TAIL window (recent CoT/answer) given [MQ][SP][recall][window-prefix].
  Loss on the window tokens only. pooler frozen; LoRA on the LLM. This closes the
  'model never trained to generate with recall' gap. Eval = recall needle operand_recovered.
"""
import os, json, time, random, re
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from attn_export3_torch import load_pooler
DEV="cuda" if torch.cuda.is_available() else "cpu"; HERE=os.path.dirname(os.path.abspath(__file__))
RW=int(os.environ.get("RW","384")); N=int(os.environ.get("N","400")); EP=int(os.environ.get("EP","1"))
def log(*a): print(*a,flush=True)

def build():
    tok=AutoTokenizer.from_pretrained("fft_hf")
    dt=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    try: m=AutoModelForCausalLM.from_pretrained("fft_hf",torch_dtype=dt,attn_implementation="eager")
    except TypeError: m=AutoModelForCausalLM.from_pretrained("fft_hf",dtype=dt,attn_implementation="eager")
    m=m.to(DEV); pooler=load_pooler("fft_out/pooler.pt").to(DEV).eval()
    for p in pooler.parameters(): p.requires_grad=False
    return tok,m,pooler

def main():
    t0=time.time(); tok,m,pooler=build(); bos=tok.bos_token_id; eos=tok.eos_token_id
    embT=m.get_input_embeddings(); log(f"[built {time.time()-t0:.0f}s] RW={RW} N={N} EP={EP}")
    # Dolphin-R1: problem + its reasoning CoT (already generated -> targets, no teacher gen)
    ds=load_dataset("mlabonne/dolphin-r1-deepseek",split="train",streaming=True)
    exs=[]
    for r in ds:
        msgs=r.get("messages") or []
        u=next((x.get("content") for x in msgs if x.get("role")=="user"),None)
        a=next((x.get("content") for x in msgs if x.get("role")=="assistant"),None)
        if u and a and 30<len(u)<800 and 200<len(a)<3000: exs.append((u,a))
        if len(exs)>=N: break
    log(f"[data] {len(exs)} Dolphin examples")
    lora=LoraConfig(r=16,lora_alpha=32,lora_dropout=0.05,bias="none",task_type="CAUSAL_LM",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])
    m=get_peft_model(m,lora); m.print_trainable_parameters(); m.config.use_cache=False
    BLOCK=128
    def make(u,a):
        head=[bos]+tok.encode(f"<｜User｜>{u}<｜Assistant｜>",add_special_tokens=False)
        cot=tok.encode(a,add_special_tokens=False)
        seq=head+cot
        if len(seq)<=RW+BLOCK: return None                     # need real eviction
        nev=len(seq)-RW                                         # evicted head
        kept=seq[:nev]; window=seq[nev:]                       # window = tail (target region)
        # recall block = the problem block (head), capped to BLOCK tokens, NL-wrapped (helped in A/B)
        rb=tok.encode('(Recalled from earlier: "',add_special_tokens=False)+head[1:1+BLOCK]+tok.encode('")',add_special_tokens=False)
        return kept,window,rb
    opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=1e-4)
    data=[d for d in (make(u,a) for u,a in exs) if d]; log(f"[usable] {len(data)} (evict>0)")
    m.train()
    for ep in range(EP):
        random.Random(ep).shuffle(data); tot=0.0; nb=0
        for kept,window,rb in data:
            with torch.no_grad():
                sp=pooler(embT(torch.tensor([kept],device=DEV)).float()).to(embT.weight.dtype)  # [1,nsp,H]
            rec=embT(torch.tensor([rb],device=DEV))
            win=embT(torch.tensor([window],device=DEV))
            inp=torch.cat([sp,rec,win],1)                       # [MQ implicit via BOS in kept? no] 
            out=m(inputs_embeds=inp)
            logits=out.logits
            # labels: only the window region (predict next token within window)
            nctx=sp.shape[1]+rec.shape[1]
            wl=len(window)
            lg=logits[0, nctx:nctx+wl-1, :]                      # logits[nctx+j-1] predicts window[j]; window[1:] => indices nctx..nctx+wl-2
            tgt=torch.tensor(window[1:],device=DEV)
            loss=F.cross_entropy(lg.float(),tgt)
            loss.backward(); torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad],1.0)
            opt.step(); opt.zero_grad(); tot+=loss.item(); nb+=1
            if nb%50==0: log(f"  ep{ep+1} {nb}/{len(data)} loss={tot/nb:.4f} t={time.time()-t0:.0f}s")
        log(f"[ep {ep+1}/{EP}] loss={tot/max(nb,1):.4f} t={time.time()-t0:.0f}s")
    m.save_pretrained(os.path.join(HERE,"recall_lora")); log("[saved recall_lora]")
    open(os.path.join(HERE,".done"),"w").write("ok"); log(f"TRAIN_DONE t={time.time()-t0:.0f}s")

if __name__=="__main__": main()
