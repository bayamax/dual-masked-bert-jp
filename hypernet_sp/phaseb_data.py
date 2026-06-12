"""phaseb_data.py — synthetic conversation generator for Phase B indexer training.

Each conversation: chit-chat filler with `n_needles` casually-mentioned exact facts,
padded to ~`target` tokens, plus recall questions with known gold values and gold
block positions. Same stitch format as the runtime stream.
"""
import random

NEEDLE_TEMPLATES = [
    ("By the way, the wifi password at the {place} was {code}.",
     "What was the wifi password at the {place}?"),
    ("I finally reached {person} today - the office extension is {num4}.",
     "What is {person}'s office extension?"),
    ("The {thing} prototype came in at {kg} kg on the bench scale.",
     "How much did the {thing} prototype weigh?"),
    ("My cousin named his fishing boat the {name}-Maru, of all things.",
     "What is my cousin's fishing boat called?"),
    ("Parking was a maze - we ended up in spot {letter}-{num2}.",
     "Which parking spot did we end up in?"),
    ("That limited edition sake cost {num4} yen, worth every drop.",
     "How much did the limited edition sake cost?"),
    ("The night train to {city} leaves at {time}, I double checked.",
     "What time does the night train to {city} leave?"),
    ("The projector in room {letter} is an XV-{num3}, in case anyone asks.",
     "What model is the projector in room {letter}?"),
    ("Grandma's bread recipe needs exactly {num3} grams of flour.",
     "How many grams of flour does grandma's bread recipe need?"),
    ("We turned back at trail marker {num2} because of the fog.",
     "At which trail marker did we turn back?"),
    ("My locker combination at the gym is {code2}.",
     "What is my locker combination at the gym?"),
    ("The vet said {pet} now weighs {kg} kilos.",
     "How much does {pet} weigh now?"),
    ("Room {num3} on the {ord} floor is where the workshop happens.",
     "Which room is the workshop in?"),
    ("The ferry ticket was {num4} yen round trip.",
     "How much was the ferry ticket?"),
    ("Dr. {person2}'s clinic moved to {num2} {street} Street.",
     "Where did Dr. {person2}'s clinic move to?"),
]

HELDOUT_TEMPLATES = [
    ("The shed door code out back is {code2}, same as last summer.",
     "What is the code for the shed door?"),
    ("Our flight back got moved to gate {letter}{num2}.",
     "Which gate is our flight back leaving from?"),
    ("The orchestra tickets were row {letter}, seats {num2} and after.",
     "Which row are the orchestra tickets in?"),
    ("The mechanic quoted {num4} yen to replace the brake pads.",
     "How much did the mechanic quote for the brake pads?"),
    ("My library card number ends in {num3}, oddly easy to remember.",
     "What does my library card number end in?"),
]

FILLER = [
    ("The weather this week has been all over the place, sunny then storms.",
     "It really has. Those sudden shifts make it hard to plan anything outdoors, though the cool evenings after the rain have been pleasant for walks."),
    ("I tried that new ramen place near the station.",
     "How was it? A good bowl of ramen after a long day is hard to beat - the broth makes or breaks it in my experience."),
    ("Work has been busy but manageable lately.",
     "That sounds like a decent balance. Busy enough to stay engaged without burning out is the sweet spot most people aim for."),
    ("I started reading a novel about Antarctic explorers.",
     "Polar exploration stories can be gripping - the isolation and the small daily decisions that decide everything give them real tension."),
    ("My neighbor's dog learned to open the gate somehow.",
     "Clever dog! Once they figure out latches there is no stopping them - you may need a second clip higher up where paws cannot reach."),
    ("We are thinking about repainting the kitchen.",
     "A fresh coat changes the whole feel of a room. Lighter colors tend to make small kitchens feel bigger, especially with warm lighting."),
    ("The train was delayed twenty minutes this morning.",
     "Annoying way to start the day. At least with a book or some music the platform wait goes a little faster."),
    ("I have been trying to drink more water instead of coffee.",
     "A solid habit. Keeping a full bottle on the desk where you can see it does most of the work, honestly."),
    ("The garden tomatoes are finally turning red.",
     "Nothing beats a sun-warmed tomato straight off the vine. The first ripe one of the season always feels like a small victory."),
    ("I watched a documentary about deep sea creatures last night.",
     "The deep sea is wilder than fiction - bioluminescent lures, pressure-proof bodies, whole ecosystems around vents. What stuck with you most?"),
    ("My phone battery barely lasts the afternoon now.",
     "Batteries do fade after a couple of years. Lowering screen brightness and trimming background apps can buy you a few more months."),
    ("We played board games with friends over the weekend.",
     "Board game nights are underrated. The slow conversational pace between turns is half the fun, win or lose."),
    ("There was a small earthquake here yesterday, nothing serious.",
     "Glad it was minor. Always a good prompt to check the emergency kit and strap down the tall furniture, just in case."),
    ("I am learning to make sourdough bread.",
     "A rewarding rabbit hole. Keeping the starter happy is the hard part - consistent feeding times matter more than fancy flour."),
    ("The library extended its evening hours this month.",
     "That is great news for anyone who works late. Quiet evening hours at a library are some of the best focused time there is."),
    ("My bicycle needs a new chain, it keeps slipping.",
     "A slipping chain usually means it has stretched past its service life. Replacing it early also saves the cassette from extra wear."),
]

_PLACES = ["mountain cabin", "beach house", "ski lodge", "lake cottage", "hostel"]
_PEOPLE = ["Dr. Imai", "Professor Sato", "Ms. Tanaka", "Mr. Watanabe", "Dr. Mori"]
_THINGS = ["rover", "drone", "robot arm", "sensor rig", "gimbal"]
_NAMES = ["Yume", "Kaze", "Hikari", "Sora", "Nami"]
_CITIES = ["Aomori", "Sendai", "Niigata", "Kanazawa", "Matsumoto"]
_PETS = ["Mochi", "Kuro", "Hana", "Taro", "Momo"]
_STREETS = ["Sakura", "Willow", "Harbor", "Maple", "Chestnut"]
_ORDS = ["second", "third", "fourth", "fifth"]


def _fill(t, rng):
    return (t.replace("{place}", rng.choice(_PLACES))
             .replace("{person2}", rng.choice(_PEOPLE).split()[-1])
             .replace("{person}", rng.choice(_PEOPLE))
             .replace("{thing}", rng.choice(_THINGS))
             .replace("{name}", rng.choice(_NAMES))
             .replace("{city}", rng.choice(_CITIES))
             .replace("{pet}", rng.choice(_PETS))
             .replace("{street}", rng.choice(_STREETS))
             .replace("{ord}", rng.choice(_ORDS))
             .replace("{code2}", f"{rng.randint(10,99)}-{rng.randint(10,99)}-{rng.randint(10,99)}")
             .replace("{code}", f"{rng.choice(['trout','falcon','cedar','quartz','ember'])}-{rng.randint(1000,9999)}")
             .replace("{num4}", str(rng.randint(1000, 9999)))
             .replace("{num3}", str(rng.randint(100, 999)))
             .replace("{num2}", str(rng.randint(10, 99)))
             .replace("{kg}", f"{rng.randint(2,60)}.{rng.randint(0,9)}")
             .replace("{time}", f"{rng.randint(17,23)}:{rng.randint(10,59)}")
             .replace("{letter}", rng.choice("ABCDEF")))


def make_conversation(seed, n_needles=3, target=3800, heldout=False):
    """Returns (pairs, questions): pairs = [(user, assistant)], questions =
    [(question, needle_user_text)] where needle_user_text locates the gold block."""
    rng = random.Random(seed)
    pool = HELDOUT_TEMPLATES if heldout else NEEDLE_TEMPLATES
    tpls = rng.sample(pool, min(n_needles, len(pool)))
    fillers = rng.sample(FILLER, len(FILLER))
    pairs, questions, fi = [], [], 0
    for ut, qt in tpls:
        pairs.append(fillers[fi % len(fillers)]); fi += 1
        # one rng pass fills both with the SAME values: do it jointly
        joint = _fill(ut + "|||" + qt, rng)
        u, q = joint.split("|||")
        pairs.append((u, "Noted - thanks for telling me, that sounds like quite a day."))
        questions.append((q, u))
        pairs.append(fillers[fi % len(fillers)]); fi += 1
    lap = 0
    est = sum(len((u + a).split()) for u, a in pairs) * 1.4
    while est < target:
        u, a = fillers[fi % len(fillers)]
        if lap:
            u = "One more thing about that - " + u[0].lower() + u[1:]
        pairs.append((u, a)); fi += 1
        if fi % len(fillers) == 0:
            lap += 1
        est += len((u + a).split()) * 1.4
    return pairs, questions


def to_ids(tok, pairs):
    ids, spans = [], []                 # spans[i] = (start, end) of pair i in token stream
    for u, a in pairs:
        text = ("<｜end▁of▁sentence｜>" if ids else "") + f"<｜User｜>{u}<｜Assistant｜>{a}"
        t = tok.encode(text, add_special_tokens=False)
        spans.append((len(ids), len(ids) + len(t)))
        ids.extend(t)
    return ids, spans
