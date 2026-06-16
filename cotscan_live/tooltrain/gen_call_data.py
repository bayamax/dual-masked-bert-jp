"""gen_call_data.py — SFT the TOOL-CALL DECISION only (up to <search>).
gap -> <think>..</think><search>QUERY</search> ; known/personal/chitchat -> no search.
No Wikipedia fetch (only the call + query). Balanced classes. For no-call classes the ANSWER is
left to be self-distilled from the fft model in Phase 2 (preserve its style); here we tag them
with a {{ANSWER:...}} hint the SFT step replaces with the model's own generation.

Output: call_sft.jsonl {prompt, completion, kind, answer_hint}
"""
import json, random, os
R=random.Random(13); OUT=os.path.dirname(os.path.abspath(__file__))
GAP_ENTS=[("the Treaty of Tordesillas","signing year"),("the Peace of Westphalia","signing year"),
 ("the Eiffel Tower","completion year"),("the Suez Canal","opening year"),("Saint Petersburg","founding year"),
 ("the Krakatoa eruption","year"),("the Rosetta Stone","discovery year"),("the Magna Carta","year"),
 ("the Battle of Hastings","year"),("the Hoover Dam","completion year"),("the Berlin Wall","construction year"),
 ("the Chernobyl disaster","year"),("the Panama Canal","completion year"),("the Statue of Liberty","dedication year"),
 ("Pompeii","destruction year"),("the Great Fire of London","year"),("penicillin","discovery year"),
 ("the Sydney Opera House","opening year"),("the Antikythera mechanism","discovery year"),("the Wright Flyer","first flight year"),
 ("the Colosseum","completion year"),("the Trans-Siberian Railway","completion year"),("Carthage","founding year"),
 ("the Apollo 11 mission","year"),("the element Francium","discovery year"),("the Voyager 1 probe","launch year"),
 ("the Taj Mahal","completion year"),("the Gutenberg Bible","printing year"),("Angkor Wat","construction era"),("the Hagia Sophia","completion year")]
GAP_Q=["When was {e} {a}? Look it up.","Tell me the {a} of {e}.","I need the {a} for {e}.",
 "Could you check the {a} of {e}?","What's the {a} of {e}?"]
GAP_LIVE=["What is today's price of crude oil?","What's the current population of Tokyo?","Who is the current CEO of Sony?",
 "What's the latest yen to dollar exchange rate?","What's the current marathon world record?","What is the population of Canberra?",
 "Who is the current UN Secretary-General?","What's the height of the Burj Khalifa?","What's the latest iPhone model?","Who won the most recent Super Bowl?"]
KNOWN=[("What is the capital of France?","Paris."),("What is the chemical symbol for gold?","Au."),
 ("Who wrote Romeo and Juliet?","William Shakespeare."),("Which planet is the Red Planet?","Mars."),
 ("What is the largest ocean on Earth?","The Pacific Ocean."),("What is the capital of Japan?","Tokyo."),
 ("How many continents are there?","Seven."),("What is the boiling point of water in Celsius?","100 degrees."),
 ("Who painted the Mona Lisa?","Leonardo da Vinci."),("What language is spoken in Brazil?","Portuguese."),
 ("What is 12 times 11?","132."),("What is the capital of Italy?","Rome."),("How many sides does a triangle have?","Three."),
 ("What is the freezing point of water in Celsius?","0 degrees."),("Who developed the theory of relativity?","Albert Einstein."),
 ("What is the capital of Germany?","Berlin."),("What gas do humans breathe in to survive?","Oxygen."),("What is H2O commonly known as?","Water.")]
PERSONAL_FACTS=[("flight number","JL412"),("locker code","C12"),("trip budget","47000 dollars"),("project codename","Bluefin"),
 ("hotel confirmation number","88231"),("car","a blue Subaru"),("dog's name","Mochi"),("home city","Kyoto"),
 ("favorite color","teal"),("seat number","14C")]
PERSONAL_Q=["What was my {k} again?","Remind me my {k}?","Can you tell me my {k}?","What's my {k}?"]
CHITCHAT=[("I just adopted two kittens.","That's wonderful! Kittens bring so much joy. How are they settling in?"),
 ("The weather has been lovely lately.","Glad to hear it! Nice weather really lifts the mood."),
 ("I'm a bit stressed about a deadline.","That sounds tough. Breaking it into small steps can help — want to talk it through?"),
 ("I tried a great ramen place yesterday.","Nice! A good bowl of ramen is hard to beat. What did you order?"),
 ("Hey, how's it going?","Hi! I'm doing well and happy to help. What's on your mind?"),
 ("I love hiking on weekends.","Hiking is a great way to recharge. Any favorite trails?"),
 ("I'm feeling pretty good today.","That's great to hear! Anything fun planned?"),
 ("Just finished a long week at work.","You earned a break! Hope you can relax this weekend."),
 ("I started learning the guitar.","Awesome — the first chords are the hardest. What song are you aiming for?"),
 ("My neighbor got a new puppy.","Aww, puppies are the best. Bet your street just got more adorable."),
 ("I might take a short trip next month.","Sounds refreshing! Anywhere in mind?"),
 ("Good morning!","Good morning! Hope your day's off to a good start."),
 ("I watched a great movie last night.","Nice! What did you watch?"),
 ("Thanks for your help earlier.","Anytime — glad I could help!")]
def think(s): return f"<think>{s}</think>"
TG=think("This needs a specific/current fact I shouldn't guess; I'll search.")
def main():
    rows=[]; add=lambda p,c,k,h=None: rows.append({"prompt":p,"completion":c,"kind":k,"answer_hint":h})
    for e,a in GAP_ENTS:
        for q in R.sample(GAP_Q,2):
            add(f"<｜User｜>{q.format(e=e,a=a)}<｜Assistant｜>", f"{TG}<search>{e} {a}</search>","gap")
    for q in GAP_LIVE:
        term=q.rstrip('?').replace('What is ','').replace("What's ",'').replace('Who is ','').replace('Who won ','')
        add(f"<｜User｜>{q}<｜Assistant｜>", f"{TG}<search>{term}</search>","gap")
    # no-call: completion's answer will be self-distilled in Phase 2; store hint
    for q,a in KNOWN:
        add(f"<｜User｜>{q}<｜Assistant｜>", f"{think('I know this; no search needed.')}{a}","known",a)
    for k,v in PERSONAL_FACTS:
        ctx=f"(Earlier the user said their {k} is {v}.)"
        for q in R.sample(PERSONAL_Q,2):
            add(f"{ctx}<｜User｜>{q.format(k=k)}<｜Assistant｜>", f"{think('About the user; in memory, no search.')}{v}.","personal",v)
    for u,a in CHITCHAT:
        add(f"<｜User｜>{u}<｜Assistant｜>", f"{think('Casual; reply warmly, no search.')}{a}","chitchat",a)
    R.shuffle(rows)
    open(os.path.join(OUT,"call_sft.jsonl"),"w").write("\n".join(json.dumps(r) for r in rows))
    from collections import Counter
    c=Counter(r['kind'] for r in rows)
    print("call_sft pairs:",len(rows),dict(c),"| gap:",c['gap'],"no-call:",len(rows)-c['gap'])
if __name__=="__main__": main()
