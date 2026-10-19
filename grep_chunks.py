import json, sys

kw = sys.argv[1]
frag = sys.argv[2] if len(sys.argv) > 2 else ""
width = int(sys.argv[3]) if len(sys.argv) > 3 else 120
found = 0
for line in open("data/chunks.jsonl", encoding="utf-8"):
    r = json.loads(line)
    if kw in r["text"] and frag in r["url"]:
        i = r["text"].find(kw)
        print(r["url"])
        print("   ...", r["text"][max(0, i - 80): i + width].replace("\n", " "))
        found += 1
print("\nנמצאו:", found)