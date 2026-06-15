import os, sys, re
BASE=os.path.join(os.path.dirname(os.path.abspath(__file__)), "faithful_run")
sys.path.insert(0, BASE); sys.path.insert(0, os.path.join(BASE,"runtime")); os.chdir(BASE)
import joblib
from rag import BGERetriever
bge=BGERetriever(); spec=joblib.load("evals/specificity_clf.joblib")
_CAND = re.compile(r"\$\d[\d,]*(?:\.\d+)?|\b\d{1,2}(?::\d{2})?\s?(?:am|pm)\b|\b\d+(?:\.\d+)?\s?(?:cm|mm|km|kg|%|percent|days?|years?|hours?|min)\b|\b[A-Za-z]*\d[A-Za-z0-9]*(?:-[A-Za-z0-9]+)*\b|\b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,}){0,2}\b",0)
def spans(text,min_p=0.6):
    cands=[c for c in dict.fromkeys(m.group(0).strip() for m in _CAND.finditer(text)) if len(c)>=2][:24]
    if not cands: return []
    X=bge._encode(cands,is_query=False); si=list(spec["clf"].classes_).index(1)
    p=spec["clf"].predict_proba(X)[:,si]
    return sorted(((c,round(float(pp),2)) for c,pp in zip(cands,p) if pp>=min_p), key=lambda h:-h[1])
for t in ["Hey! I'm planning a trip next month.",
 "I've been so busy lately, work has been hectic with back-to-back meetings.",
 "The weather has been really nice though, lots of sunshine.",
 "I tried a new ramen place yesterday, it was delicious.",
 "Please remember: my flight number is JL412.",
 "Also my budget cap is 47000 dollars."]:
    print(f"spans={spans(t)} | {t}")
