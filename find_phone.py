import json
import re
from collections import defaultdict

PHONE = re.compile(r"(\*\d{3,4}|\b0\d{1,2}[-\s]?\d{7}\b|\b1[-\s]?[78]\d{2}[-\s]?\d{3}[-\s]?\d{3}\b)")
WORDS = ("שירות", "מוקד", "טלפון", "התקשר", "חייג", "פנייה", "פניה")

found = defaultdict(list)
with open("data/pages.jsonl", encoding="utf-8") as f:
    for line in f:
        d = json.loads(line)
        text = d.get("text", "")
        for m in PHONE.finditer(text):
            ctx = text[max(0, m.start() - 70): m.end() + 40].replace("\n", " ")
            if any(w in ctx for w in WORDS):
                found[m.group(1)].append((d["url"], ctx))

for num, items in sorted(found.items(), key=lambda kv: -len(kv[1]))[:8]:
    print(f"\n=== {num}   (מופיע ב-{len(items)} מקומות)")
    for url, ctx in items[:2]:
        print("   ", url)
        print("   ...", ctx)