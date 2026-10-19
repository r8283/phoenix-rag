
import json
from pathlib import Path

from chunking import chunk_page
from config import CHUNK_OVERLAP_CHARS

pages = [
    json.loads(line)
    for line in Path("data/pages.jsonl")
    .read_text(encoding="utf-8")
    .splitlines()
    if line.strip()
]

problems = []
checked_transitions = 0
overlaps = []

for page_index, page in enumerate(pages):
    chunks = chunk_page(
        page["url"],
        page.get("title") or "",
        page.get("text") or "",
        page.get("fetched_at") or "",
    )

    for chunk_index, chunk in enumerate(chunks):
        text = chunk["text"]

        if not text:
            problems.append((page_index, chunk_index, "קטע ריק"))
        elif text != text.strip():
            problems.append(
                (page_index, chunk_index, "רווח בתחילת הקטע או בסופו")
            )

        if len(text) > 1200:
            problems.append(
                (page_index, chunk_index, f"קטע ארוך מדי: {len(text)}")
            )

    for i in range(len(chunks) - 1):
        first = chunks[i]["text"]
        second = chunks[i + 1]["text"]

        # Measure the longest suffix of the first chunk
        # that matches the beginning of the next chunk.
        max_possible = min(len(first), len(second))
        overlap = 0

        for size in range(max_possible, 0, -1):
            if first[-size:] == second[:size]:
                overlap = size
                break

        checked_transitions += 1
        overlaps.append(overlap)

        if overlap == 0:
            problems.append(
                (page_index, i, "לא נמצאה חפיפה בין קטעים סמוכים")
            )
        elif not (CHUNK_OVERLAP_CHARS - 50 <= overlap <= CHUNK_OVERLAP_CHARS + 20):
            problems.append(
                (page_index, i, f"חפיפה חריגה: {overlap} תווים")
            )

print("מספר דפים:", len(pages))
print("מעברים בין קטעים שנבדקו:", checked_transitions)
print("מספר מעברים עם חפיפה אפס:", sum(x == 0 for x in overlaps))
if overlaps:
    print("חפיפה מינימלית:", min(overlaps))
    print("חפיפה מקסימלית:", max(overlaps))
    print("חפיפה ממוצעת:", round(sum(overlaps) / len(overlaps), 1))
print("מספר בעיות שזוהו:", len(problems))
print("דוגמאות לבעיות:", problems[:10])
