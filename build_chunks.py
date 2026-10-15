
import json
from pathlib import Path

from chunking import pages_to_chunks

pages_path = Path("data/pages.jsonl")
output_path = Path("data/chunks.jsonl")

pages = [
    json.loads(line)
    for line in pages_path.read_text(encoding="utf-8").splitlines()
    if line.strip()
]

chunks = pages_to_chunks(pages)

with output_path.open("w", encoding="utf-8") as file:
    for chunk in chunks:
        file.write(json.dumps(chunk, ensure_ascii=False) + "\n")

print("מספר דפים:", len(pages))
print("מספר קטעים שנשמרו:", len(chunks))
print("קובץ הפלט:", output_path)
print("גודל הקובץ בבתים:", output_path.stat().st_size)
