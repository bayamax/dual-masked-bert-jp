"""gen_tool_data.py — Phase 1 (Wikipedia, robust): tool-call <search> training data with REAL
Wikipedia results and CURATED golds. For each gap topic we fetch the full intro extract and use
the sentence that actually CONTAINS the curated gold as the <results> snippet -> grounding is real
and the gold is correct (no mislabeled not-found). Genuine not-found uses nonexistent topics.

Outputs: cold_start_sft.jsonl {prompt,completion,kind} | grpo_queries.jsonl {query,ctx,should_call,gold,results,kind}
"""
import json, random, os, re, ssl, time, urllib.request, urllib.parse
R=random.Random(7); OUT=os.path.dirname(os.path.abspath(__file__))
CTX=ssl._create_unverified_context(); UA="sp-tooltrain/1.0"
def _get(u):
    last=None
    for a in range(4):
        try: return json.load(urllib.request.urlopen(urllib.request.Request(u,headers={"User-Agent":UA}),timeout=25,context=CTX))
        except Exception as e: last=e; time.sleep(1.5*(a+1))
    raise last
def wiki_top(q):
    u="https://en.wikipedia.org/w/api.php?"+urllib.parse.urlencode({"action":"query","list":"search","srsearch":q,"format":"json","srlimit":1})
    h=_get(u)["query"]["search"]; return h[0]["title"] if h else None
def wiki_extract(title):
    u="https://en.wikipedia.org/w/api.php?"+urllib.parse.urlencode(
        {"action":"query","prop":"extracts","exintro":1,"explaintext":1,"titles":title,"format":"json","redirects":1})
    pgs=_get(u)["query"]["pages"]; return next(iter(pgs.values())).get("extract","")
def sent_with(text, gold):
    for s in re.split(r"(?<=[.!?])\s+", text):
        if gold.lower() in s.lower(): return " ".join(s.split())
    return None

# (question, wikipedia term, CURATED gold) — gold is the correct answer; we keep only items whose
# real Wikipedia intro contains the gold (guarantees grounded, correct training signal).
GAP = [
  ("In what year was the Treaty of Tordesillas signed?","Treaty of Tordesillas","1494"),
  ("In what year was the Peace of Westphalia signed?","Peace of Westphalia","1648"),
  ("In what year was the Eiffel Tower completed?","Eiffel Tower","1889"),
  ("In what year did the Suez Canal open?","Suez Canal","1869"),
  ("In what year was Saint Petersburg founded?","Saint Petersburg","1703"),
  ("In what year did the Krakatoa eruption occur?","1883 eruption of Krakatoa","1883"),
  ("In what year was the Rosetta Stone discovered?","Rosetta Stone","1799"),
  ("In what year was the Magna Carta sealed?","Magna Carta","1215"),
  ("In what year did the Battle of Hastings take place?","Battle of Hastings","1066"),
  ("In what year was the Hoover Dam completed?","Hoover Dam","1936"),
  ("In what year was the Berlin Wall built?","Berlin Wall","1961"),
  ("In what year did the Chernobyl disaster happen?","Chernobyl disaster","1986"),
  ("In what year was the Panama Canal completed?","Panama Canal","1914"),
  ("In what year was the Statue of Liberty dedicated?","Statue of Liberty","1886"),
  ("In what year did Mount Vesuvius destroy Pompeii?","Pompeii","79"),
  ("In what year was the Great Fire of London?","Great Fire of London","1666"),
  ("In what year was the transcontinental railroad completed in the US?","First transcontinental railroad","1869"),
  ("In what year was the Sydney Opera House opened?","Sydney Opera House","1973"),
  ("In what year was penicillin discovered by Fleming?","Penicillin","1928"),
  ("In what year did the Wright brothers' first flight occur?","Wright Flyer","1903"),
]
NOTFOUND_TOPICS = [  # genuinely nonexistent -> honest miss
  ("In what year was the Treaty of Velmaran signed?","Treaty of Velmaran"),
  ("In what year was the city of Quorvath founded?","city of Quorvath"),
  ("Who won the Galvin Prize in 2019?","Galvin Prize 2019"),
  ("How tall is Mount Drennakar?","Mount Drennakar"),
]
KNOWN=[("What is the capital of France?","Paris"),("What is the chemical symbol for gold?","Au"),
  ("Who wrote Romeo and Juliet?","Shakespeare"),("Which planet is the Red Planet?","Mars"),
  ("What is the largest ocean on Earth?","the Pacific Ocean"),("What is the capital of Japan?","Tokyo"),
  ("How many continents are there?","seven"),("What is the boiling point of water in Celsius?","100"),
  ("Who painted the Mona Lisa?","Leonardo da Vinci"),("What language is spoken in Brazil?","Portuguese"),
  ("What is the square root of 144?","12"),("What is the capital of Italy?","Rome")]
PERSONAL=[("my flight number is JL412","What was my flight number again?","JL412"),
  ("my locker code is C12","What's my locker code?","C12"),
  ("my budget for the trip is 47000 dollars","Remind me my trip budget?","47000"),
  ("my project codename is Bluefin","What's my project codename?","Bluefin"),
  ("my hotel confirmation number is 88231","What was my hotel confirmation number?","88231")]
CHITCHAT=[("I just adopted two kittens.","That's wonderful! Kittens bring so much joy. How are they settling in?"),
  ("The weather has been lovely lately.","Glad to hear it! Nice weather is a great mood lifter."),
  ("I'm a bit stressed about a deadline.","That's tough. Breaking the work into small steps can help."),
  ("I tried a great ramen place yesterday.","Nice! A good bowl of ramen is hard to beat. What did you order?"),
  ("Hey, how's it going?","Hi! I'm doing well and happy to help. What's on your mind?"),
  ("I love hiking on weekends.","Hiking is a great way to recharge. Any favorite trails?")]
TG="<think>This is a specific factual lookup I shouldn't guess at; I'll check Wikipedia.</think>"
def rb(s): return "<results>\n- "+s+"\n</results>"

def main():
    sft,grpo=[],[]; add=lambda p,c,k: sft.append({"prompt":p,"completion":c,"kind":k})
    kept=0; dropped=0
    for q,term,gold in GAP:
        t=None; ex=""
        try:
            t=wiki_top(term); ex=wiki_extract(t) if t else ""
        except Exception as e:
            print("  fetch err:",str(e)[:50])
        time.sleep(1.0)
        snip=sent_with(ex, gold) if ex else None
        if snip:
            snip=(t+": "+snip)[:320]
            add(f"<｜User｜>{q}<｜Assistant｜>",f"{TG}<search>{term}</search>","gap_call")
            add(f"<｜User｜>{q}<｜Assistant｜>{TG}<search>{term}</search>{rb(snip)}",f"Based on Wikipedia, {gold}.","gap_ground")
            grpo.append({"query":q,"ctx":"","should_call":1,"gold":gold,"results":[snip],"kind":"gap"}); kept+=1
            print(f"[gap OK] {gold:>5}  {q[:42]}")
        else:
            dropped+=1; print(f"[drop] gold not in intro: {q[:42]} (title={t})")
    for q,term in NOTFOUND_TOPICS:
        add(f"<｜User｜>{q}<｜Assistant｜>{TG}<search>{term}</search>{rb('No matching Wikipedia article was found.')}",
            "I couldn't find reliable information on that.","notfound")
        grpo.append({"query":q,"ctx":"","should_call":1,"gold":None,"results":["No matching Wikipedia article was found."],"kind":"notfound"})
    for q,a in KNOWN:
        add(f"<｜User｜>{q}<｜Assistant｜>",f"<think>I know this from general knowledge; no search.</think>{a}.","known")
        grpo.append({"query":q,"ctx":"","should_call":0,"gold":a,"results":None,"kind":"known"})
    for fact,q,gold in PERSONAL:
        ctx=f"(Earlier the user said: {fact}.)"
        add(f"{ctx}<｜User｜>{q}<｜Assistant｜>",f"<think>About the user; in memory.</think>{gold}.","personal")
        grpo.append({"query":q,"ctx":ctx,"should_call":0,"gold":gold,"results":None,"kind":"personal"})
    for u,a in CHITCHAT:
        add(f"<｜User｜>{u}<｜Assistant｜>",f"<think>Casual; reply warmly, no search.</think>{a}","chitchat")
        grpo.append({"query":u,"ctx":"","should_call":0,"gold":None,"results":None,"kind":"chitchat"})
    open(os.path.join(OUT,"cold_start_sft.jsonl"),"w").write("\n".join(json.dumps(r) for r in sft))
    open(os.path.join(OUT,"grpo_queries.jsonl"),"w").write("\n".join(json.dumps(r) for r in grpo))
    from collections import Counter
    print(f"\ngap kept={kept} dropped={dropped}")
    print("SFT:",len(sft),dict(Counter(r['kind'] for r in sft)))
    print("GRPO:",len(grpo),dict(Counter(r['kind'] for r in grpo)))
    g=next(r for r in sft if r['kind']=='gap_ground')
    print("\n[gap_ground]\n P:",g['prompt'][:300],"\n C:",g['completion'])

if __name__=="__main__": main()
