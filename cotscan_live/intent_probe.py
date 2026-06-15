import os, sys
BASE=os.path.join(os.path.dirname(os.path.abspath(__file__)), "faithful_run")
sys.path.insert(0, BASE); sys.path.insert(0, os.path.join(BASE,"runtime")); os.chdir(BASE)
import joblib
from rag import BGERetriever
from intent_route import route_intent, regex_intent
bge=BGERetriever(); iclf=joblib.load("evals/intent_clf.joblib")
tests=["Hey! I'm planning a trip next month.",
 "I've been so busy lately, work has been hectic with back-to-back meetings.",
 "The weather has been really nice though, lots of sunshine.",
 "I tried a new ramen place yesterday, it was delicious.",
 "Please remember: my flight number is JL412.",
 "What's the capital of France?",
 "What is 47000 divided by 30?",
 "Can you recommend something fun to do on a weekend trip?",
 "What was my flight number again?"]
import numpy as np
for t in tests:
    r=route_intent(t, iclf, bge); rx=regex_intent(t)
    # also raw clf top probs
    try:
        X=bge._encode([t], is_query=True); p=iclf.predict_proba(X)[0]; cls=iclf.classes_
        top=sorted(zip(cls,p), key=lambda z:-z[1])[:3]
        tops=" ".join(f"{c}:{pr:.2f}" for c,pr in top)
    except Exception as e: tops=f"(noproba {e})"
    print(f"route={r:9} regex={rx:9} | clf[{tops}] | {t}")
