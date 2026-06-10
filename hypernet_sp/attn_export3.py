"""STEP1 (MLX), scaled (STATUS 2026-06-10, next-step 1): 30 scenario triples instead of 5.
For each scenario, compress turn-1 into SP and tokenize a REFERENTIAL turn-2 (needs a turn-1
value, which therefore lives ONLY in the SP) and a matched CONTROL turn-2 (self-contained —
the value is in the question, so the SP isn't needed). Same SP for both. Categories cover
arithmetic chains, codes/IDs, names, schedule, preferences and measurements so the probe's
heads aren't an artifact of one task family. Save for attn_probe3.py.

Run next to sp_mlx.py in the HF repo:  python3 attn_export3.py
"""
import numpy as np, mlx.core as mx
import sp_mlx

M = sp_mlx.get()
tok, pooler, embT = M["tok"], M["pooler"], M["embT"]
emb = lambda ids: embT(mx.array([ids]))

# (category, turn-1 fact, referential turn-2 [omits the needed value], control turn-2 [includes it])
SCEN = [
 # -- arithmetic chains (the original 5 kept verbatim for comparability) --
 ("math", "A bakery sells muffins for $4 each. Maria buys 6 muffins. How much does she spend?",
  "I pay with a $50 bill. How much change do I get back?",
  "A toy costs $30 and I pay with a $50 bill. How much change do I get back?"),
 ("math", "Tom reads 15 pages each night for 8 nights. How many pages does he read in total?",
  "If the book has 200 pages, how many pages are left to read?",
  "A book has 200 pages and 120 are already read. How many pages are left to read?"),
 ("math", "A class has 30 students. 40% of them are boys. How many boys are there?",
  "How many of the students are girls?",
  "A class has 30 students and 12 are boys. How many are girls?"),
 ("math", "Jake earns $12 per hour and works 9 hours. How much does he earn?",
  "If he saves half of what he earned, how much does he save?",
  "Jake earned $108. If he saves half of it, how much does he save?"),
 ("math", "Sara has 5 boxes with 24 pencils in each box. How many pencils total?",
  "She gives away 45 pencils. How many does she have left?",
  "Sara has 120 pencils and gives away 45. How many does she have left?"),
 ("math", "A train travels 80 km per hour for 3 hours. How far does it go?",
  "If the whole route is 400 km, how much farther must it travel?",
  "A route is 400 km and a train has covered 240 km. How much farther must it travel?"),
 ("math", "Lena buys 4 notebooks at $7 each. How much does she pay?",
  "She had $100 in her wallet. How much does she have now?",
  "Lena had $100 and spent $28. How much does she have now?"),
 ("math", "A tank holds 600 liters and is 25% full. How many liters are in it?",
  "How many more liters are needed to fill it completely?",
  "A tank holds 600 liters and contains 150. How many more liters fill it?"),
 ("math", "A recipe needs 3 eggs per cake. Ben bakes 7 cakes. How many eggs does he use?",
  "Eggs come in boxes of 12. How many boxes does he need?",
  "Ben needs 21 eggs and they come in boxes of 12. How many boxes does he need?"),
 ("math", "Mia runs 6 km every morning for 12 days. How many km does she run?",
  "Her goal is 100 km. How many km short of the goal is she?",
  "Mia ran 72 km and her goal is 100 km. How many km short is she?"),
 # -- codes / IDs (verbatim tokens, the SP's known weak point — does it still SIGNAL?) --
 ("code", "My insurance policy number is POL-55821.",
  "What is my insurance policy number?",
  "My policy number is POL-55821. What is my insurance policy number?"),
 ("code", "The wifi password for the cabin is hunter2.",
  "What is the wifi password?",
  "The wifi password is hunter2. What is the wifi password?"),
 ("code", "My gym locker combination is 8042.",
  "What is my locker combination?",
  "My locker combination is 8042. What is my locker combination?"),
 ("code", "The reservation is booked under code QX7-2291.",
  "What code is the reservation under?",
  "The reservation code is QX7-2291. What code is the reservation under?"),
 ("code", "My employee ID is EMP-90832.",
  "What is my employee ID?",
  "My employee ID is EMP-90832. What is my employee ID?"),
 ("code", "I parked on level B3, spot 47.",
  "Which level and spot did I park in?",
  "I parked on level B3, spot 47. Which level and spot did I park in?"),
 # -- names / entities --
 ("name", "My emergency contact is my sister Mei.",
  "Who is my emergency contact?",
  "My emergency contact is my sister Mei. Who is my emergency contact?"),
 ("name", "The project is called Apollo.",
  "What is the project called?",
  "The project is called Apollo. What is the project called?"),
 ("name", "My dog is named Mochi.",
  "What is my dog's name?",
  "My dog is named Mochi. What is my dog's name?"),
 ("name", "The party is for my friend Mia.",
  "Who is the party for?",
  "The party is for my friend Mia. Who is the party for?"),
 ("name", "I'm staying at the Hilton in Sapporo.",
  "Which hotel am I staying at?",
  "I'm staying at the Hilton. Which hotel am I staying at?"),
 ("name", "The new intern's name is Daniel Okafor.",
  "What is the new intern's name?",
  "The intern is Daniel Okafor. What is the new intern's name?"),
 # -- schedule / time --
 ("time", "My flight leaves at 3pm on Friday.",
  "What time does my flight leave?",
  "My flight leaves at 3pm. What time does my flight leave?"),
 ("time", "The meeting moved to Friday at 10am.",
  "When is the meeting now?",
  "The meeting is Friday at 10am. When is the meeting now?"),
 ("time", "The deadline is next Monday.",
  "When is the deadline?",
  "The deadline is next Monday. When is the deadline?"),
 ("time", "My dentist appointment is on the 15th at 9am.",
  "When is my dentist appointment?",
  "My appointment is the 15th at 9am. When is my dentist appointment?"),
 # -- preferences / attributes --
 ("attr", "I'll paint the bookshelf matte black with 6 shelves.",
  "What color am I painting it and how many shelves?",
  "The bookshelf is matte black with 6 shelves. What color is it and how many shelves?"),
 ("attr", "I'm vegetarian and allergic to peanuts.",
  "Is there any food I should avoid?",
  "I'm vegetarian and allergic to peanuts. Is there any food I should avoid?"),
 # -- measurements --
 ("meas", "Each shelf is 80 cm wide and 30 cm deep.",
  "How wide is each shelf?",
  "Each shelf is 80 cm wide. How wide is each shelf?"),
 ("meas", "My room is 4 meters by 5 meters.",
  "What is the area of my room in square meters?",
  "A room is 4 meters by 5 meters. What is its area in square meters?"),
]

out = {"n": np.array([len(SCEN)]),
       "bos": np.array([tok.bos_token_id if tok.bos_token_id is not None else tok.encode("")[0]]),
       "cat": np.array([c for c, *_ in SCEN])}
for i, (cat, t1, ref, ctrl) in enumerate(SCEN):
    sp = pooler.forward(emb(tok.encode(t1, add_special_tokens=False)).astype(mx.float32))
    out[f"sp_{i}"] = np.array(sp.astype(mx.float32))[0]
    out[f"ref_{i}"] = np.array(tok.encode(ref, add_special_tokens=False))
    out[f"ctrl_{i}"] = np.array(tok.encode(ctrl, add_special_tokens=False))
np.savez("attn_probe3.npz", **out)
print(f"saved {len(SCEN)} scenarios, SP shape {out['sp_0'].shape}")
print("ATTN_EXPORT3_DONE")
