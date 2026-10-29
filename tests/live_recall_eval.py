"""Live check of memory search by meaning against a real /v1/embeddings server (not run by pytest).

Synthetic only: 200 filler captions and 30 "needle" events (captions and conversation lines) spread over 7 simulated
days, then 30 questions that paraphrase the needles. Reports the needle's rank by score (top-1, top-3) and the time a
search adds to a turn (the question's embedding plus the cosine scan), measured when the GPU is idle if --idle.

    python tests/live_recall_eval.py http://127.0.0.1:8421/v1 [--idle]
"""

import asyncio
import json
import random
import subprocess
import sys
import tempfile
import time

import httpx
import numpy as np

from voice_agent.recall import Recall

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8421/v1"
IDLE = "--idle" in sys.argv

NEEDLES = [  # (kind, text, question)
    ("seen", "A person drops a ring of keys into the blue bowl by the front door.", "where did I leave my keys?"),
    ("seen", "A person waters the three potted ferns on the window sill.", "when did I last water the plants?"),
    ("seen", "A person feeds the grey cat from a tin at the kitchen counter.", "did I feed the cat today?"),
    ("seen", "A person takes two white pills with a glass of water.", "when did I take my medicine?"),
    ("seen", "A person puts a passport into the top drawer of the desk.", "where did I put my passport?"),
    ("seen", "A person plugs a phone charger into the socket beside the bed.", "where did I leave the phone charger?"),
    ("seen", "A person stretches on a yoga mat in the living room.", "when did I last do my stretches?"),
    ("seen", "A person writes in a red notebook and closes it on the shelf.", "where is my red notebook?"),
    ("seen", "A person locks the back door and hangs the key on a hook.", "did I lock the back door?"),
    ("seen", "A person paints a small canvas of a lighthouse at the easel.", "when was I painting?"),
    ("seen", "A person repairs a bicycle tyre with a patch kit in the hallway.", "when did I fix the bike?"),
    ("seen", "A person sorts a stack of envelopes and bills on the dining table.", "when did I go through the mail?"),
    ("seen", "A person tries on a green rain jacket in front of the mirror.", "what jacket did I try on?"),
    ("seen", "A person bakes a loaf of bread and sets it on a cooling rack.", "when did I bake bread?"),
    ("seen", "A person plays a few chords on an acoustic guitar on the sofa.", "was I playing guitar earlier?"),
    ("seen", "A person packs a suitcase with folded shirts on the bed.", "when did I pack for the trip?"),
    ("seen", "A person charges a pair of wireless earbuds in their case on the desk.", "where are my earbuds?"),
    ("seen", "A person sets a timer on the oven while a pizza bakes.", "what was I cooking with the oven timer?"),
    ("seen", "A person reads a paperback novel with a bookmark in the armchair.", "where did I leave my book?"),
    ("seen", "A person wraps a gift box in striped paper with a ribbon.", "when did I wrap the present?"),
    ("you said", "We should book the train to Lisbon for the second week of May.", "what did we say about the Lisbon trip?"),
    ("you said", "The dentist moved my appointment to Thursday at four.", "when is my dentist appointment?"),
    ("you said", "Remind me that Robin's birthday dinner is on Saturday at the harbour restaurant.", "where is Robin's birthday dinner?"),
    ("you said", "I need to call the plumber about the leaking kitchen tap.", "what did I say about the plumber?"),
    ("you said", "My new password hint is the name of my first dog.", "what was my password hint?"),
    ("Boo said", "The parcel from the bookshop should arrive on Wednesday morning.", "when is my parcel arriving?"),
    ("Boo said", "Your car's insurance renews at the end of the month.", "what did you say about the car insurance?"),
    ("you said", "Let's try the new ramen place on Elm Street next Friday.", "which restaurant did we want to try?"),
    ("you said", "I left the spare house key with the neighbour at number twelve.", "who has the spare key?"),
    ("Boo said", "The recycling goes out on Tuesday nights this month.", "when does the recycling go out?"),
]
FILLER = ["A person types on a laptop at the desk.", "A person sits at the desk looking at a monitor.",
          "A person drinks from a white mug at the desk.", "An empty chair in front of a desk with a lamp.",
          "A person talks on the phone near the window.", "A person scrolls on a phone on the sofa.",
          "A person walks across the living room.", "A person stands in the kitchen.", "A dim room with the lights off.",
          "A person adjusts the desk lamp.", "A person leans back in the chair and stretches their arms.",
          "A person rubs their eyes at the desk.", "A person eats a sandwich at the desk."]


def idle(need=5):
    quiet = 0
    while quiet < need:
        cols = subprocess.run(["nvidia-smi", "dmon", "-s", "u", "-c", "1"], capture_output=True, text=True).stdout.split()
        sm = int(cols[-6]) if cols[-6].isdigit() else 100
        quiet = quiet + 1 if sm < 5 else 0
        if sm >= 5:
            time.sleep(2)


async def main():
    rng = random.Random(7)
    now = time.time()
    path = tempfile.mktemp(suffix=".jsonl")
    r = Recall({"url": URL, "dimensions": 256, "path": path, "days": 7, "top_k": 6, "timeout_ms": 2000})
    events = [(now - rng.uniform(60, 7 * 86400 - 600), "seen", rng.choice(FILLER)) for _ in range(200)]
    events += [(now - rng.uniform(60, 7 * 86400 - 600), k, t) for k, t, _ in NEEDLES]
    events.sort()
    t0 = time.perf_counter()
    for ts, kind, text in events:
        r.add(kind, text, ts)
    await r.flush()
    index_s = time.perf_counter() - t0
    assert len(r.items) == len(events), (len(r.items), len(events))
    top1 = top3 = found = 0
    misses = []
    for kind, text, q in NEEDLES:
        hits = await r.search(q) or []
        ranked = [h["text"] for h in sorted(hits, key=lambda h: -h["score"])]
        found += text in ranked
        top1 += bool(ranked) and ranked[0] == text
        top3 += text in ranked[:3]
        if not ranked or ranked[0] != text:
            misses.append({"q": q, "want": text, "got": ranked[:2]})
    if IDLE:
        idle()
    r.timeout_s = 2.0
    times = []
    for _, _, q in NEEDLES * 2:
        t = time.perf_counter()
        await r.search(q)
        times.append(1000 * (time.perf_counter() - t))
    out = {"entries": len(events), "days": 7, "questions": len(NEEDLES), "dimensions": 256,
           "top1": top1, "top3": top3, "in_hits": found, "index_s": round(index_s, 2),
           "search_ms_median": round(float(np.median(times)), 2), "search_ms_p95": round(float(np.percentile(times, 95)), 2),
           "within_40ms": sum(t <= 40 for t in times), "of": len(times), "idle_gpu": IDLE, "misses": misses}
    print(json.dumps(out, indent=1))


asyncio.run(main())
