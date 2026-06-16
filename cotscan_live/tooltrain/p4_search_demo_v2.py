"""p4_search_demo.py — end-to-end tool use on CPU: SFT'd model emits <search>, runtime executes
it against Wikipedia, injects <results>, model answers grounded. gap->search->Wikipedia->answer;
known/chitchat->direct.
"""
import os, sys, re, ssl, json, time, urllib.request, urllib.parse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
HERE=os.path.dirname(os.path.abspath(__file__)); DEV="cpu"
FFT=os.path.join(HERE,"..","faithful_run","fft_hf"); ADP=os.environ.get("ADP","/tmp/lora")
CTX=ssl._create_unverified_context(); UA="sp-tool/1.0"
def log(*a): print(*a,flush=True)
def _get(u):
    for _ in range(3):
        try: return json.load(urllib.request.urlopen(urllib.request.Request(u,headers={"User-Agent":UA}),timeout=20,context=CTX))
        except Exception: time.sleep(1.0)
    return {}
def wiki(query, n=1):
    su="https://en.wikipedia.org/w/api.php?"+urllib.parse.urlencode({"action":"query","list":"search","srsearch":query,"format":"json","srlimit":n})
    hits=_get(su).get("query",{}).get("search",[])
    if not hits: return None
    title=hits[0]["title"]
    eu="https://en.wikipedia.org/w/api.php?"+urllib.parse.urlencode({"action":"query","prop":"extracts","exintro":1,"explaintext":1,"titles":title,"redirects":1,"format":"json"})
    pgs=_get(eu).get("query",{}).get("pages",{})
    ext=next(iter(pgs.values()),{}).get("extract","") if pgs else ""
    return (title, ext[:400]) if ext else None

def main():
    tok=AutoTokenizer.from_pretrained(FFT)
    try: base=AutoModelForCausalLM.from_pretrained(FFT,torch_dtype=torch.float32)
    except TypeError: base=AutoModelForCausalLM.from_pretrained(FFT,dtype=torch.float32)
    m=PeftModel.from_pretrained(base,ADP).eval().to(DEV)
    bos=tok.bos_token_id; eos=tok.eos_token_id
    @torch.no_grad()
    def gen(ids,n):
        out=m.generate(input_ids=torch.tensor([ids],device=DEV),max_new_tokens=n,do_sample=False,pad_token_id=eos)
        return out[0,len(ids):].tolist()
    def turn(q):
        ids=[bos]+tok.encode(f"<｜User｜>{q}<｜Assistant｜>",add_special_tokens=False)
        g1=gen(ids,80); txt1=tok.decode(g1,skip_special_tokens=False)
        msrch=re.search(r"<search>(.*?)</search>", txt1, re.S)
        if not msrch:
            return ("DIRECT", tok.decode(g1,skip_special_tokens=True).strip(), None)
        query=msrch.group(1).strip()
        w=wiki(query)
        rb=f"<results>\n- {w[0]}: {w[1]}\n</results>" if w else "<results>\nNo matching Wikipedia article was found.\n</results>"
        # rebuild context up to and including </search>, then inject results, continue
        ref = (w[0]+": "+w[1]) if w else "No matching Wikipedia article was found."
        # re-prompt fresh with the retrieved reference (use the model's reading ability)
        ap=f"<｜User｜>{q}\n\n(Reference from Wikipedia: {ref})<｜Assistant｜>"
        ids2=[bos]+tok.encode(ap,add_special_tokens=False)+tok.encode("<think>\n\n</think>\n\n",add_special_tokens=False)
        g2=gen(ids2,90); ans=tok.decode(g2,skip_special_tokens=True).strip()
        return ("SEARCH", ans, {"query":query,"wiki_title":(w[0] if w else None),"results":(w[1][:120] if w else "none")})
    BATTERY=[("gap","Who won the FIFA World Cup in 1998?"),("gap","When was the Brooklyn Bridge opened?"),("gap","Who painted Guernica?")]
    LOG=[]
    def out(x): print(x,flush=True); LOG.append(x); open(os.path.join(HERE,"p4_demo2.log"),"w").write("\n".join(LOG))
    for kind,q in BATTERY:
        t=time.time(); mode,ans,meta=turn(q)
        out(f"\n[{kind}] USER> {q}")
        if meta: out(f"  -> SEARCH query={meta['query']!r} wiki={meta['wiki_title']!r}\n     results: {meta['results']}")
        out(f"  BOT({mode}, {time.time()-t:.0f}s)> {ans[:220]}")
    out("\n[DONE]")

if __name__=="__main__": main()
