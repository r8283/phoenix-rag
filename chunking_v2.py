"""
chunking_v2.py - חיתוך לפי פסקאות, כותרת בכל קטע, בלי כפילויות.

קורא data/pages.jsonl וכותב data/chunks.jsonl (באותו פורמט ש-build_embeddings.py מצפה לו).
שימוש:  python chunking_v2.py
"""
import hashlib
import json
import re
import statistics
from pathlib import Path

PAGES_PATH = Path("data/pages.jsonl")
CHUNKS_PATH = Path("data/chunks.jsonl")

MAX_CHARS = 1200          # גודל קטע מקסימלי
MIN_KEEP = 150            # קטע קצר מזה נזרק
TAIL_MERGE_BELOW = 400    # שארית קצרה מזה מצורפת לקטע הקודם
HEADING_MAX = 80          # שורה קצרה בלי סימן סיום נחשבת כותרת
EXCLUDE_URL_PARTS = ("-old/",)   # עמודים ישנים. כדי לכלול הכל: ()

SENT_SPLIT = re.compile(r"(?<=[.!?:;])\s+")


def md5(text):
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def is_heading(line):
    return len(line) <= HEADING_MAX and not re.search(r"[.!:;,]$", line)


def split_units(text):
    """מפרק טקסט ליחידות (פסקאות, או משפטים אם הפסקה ארוכה מדי)."""
    units = []
    for para in text.split("\n"):
        para = para.strip()
        if not para:
            continue
        if len(para) <= MAX_CHARS:
            units.append(para)
            continue
        buf = ""
        for sent in SENT_SPLIT.split(para):
            while len(sent) > MAX_CHARS:
                cut = sent.rfind(" ", 0, MAX_CHARS)
                cut = cut if cut > 0 else MAX_CHARS
                piece, sent = sent[:cut], sent[cut:].strip()
                if buf:
                    units.append(buf)
                    buf = ""
                units.append(piece)
            if buf and len(buf) + 1 + len(sent) > MAX_CHARS:
                units.append(buf)
                buf = sent
            else:
                buf = f"{buf} {sent}".strip()
        if buf:
            units.append(buf)
    return units


def pack_units(units):
    """אורז יחידות לקטעים. כותרת לא נשארת לבד בסוף קטע."""
    chunks, cur, cur_len = [], [], 0
    for u in units:
        if cur and cur_len + len(u) + 1 > MAX_CHARS:
            carry = []
            while len(cur) > 1 and is_heading(cur[-1]):
                carry.insert(0, cur.pop())
            chunks.append("\n".join(cur))
            cur = carry
            cur_len = sum(len(x) + 1 for x in cur)
        cur.append(u)
        cur_len += len(u) + 1
    if cur:
        chunks.append("\n".join(cur))
    if (
        len(chunks) >= 2
        and len(chunks[-1]) < TAIL_MERGE_BELOW
        and len(chunks[-2]) + len(chunks[-1]) + 1 <= MAX_CHARS * 1.4
    ):
        tail = chunks.pop()
        chunks[-1] += "\n" + tail
    return chunks


def main():
    pages = [
        json.loads(line)
        for line in PAGES_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    seen_pages, seen_chunks, out = set(), set(), []
    dup_pages = dup_chunks = short = excluded = 0

    for p_idx, page in enumerate(pages):
        if any(part in page["url"] for part in EXCLUDE_URL_PARTS):
            excluded += 1
            continue
        body = (page.get("text") or "").strip()
        h = md5(body)
        if h in seen_pages:
            dup_pages += 1
            continue
        seen_pages.add(h)

        title = (page.get("title") or page["url"]).split("|")[0].strip()
        for c_idx, chunk_body in enumerate(pack_units(split_units(body))):
            if len(chunk_body) < MIN_KEEP:
                short += 1
                continue
            ch = md5(chunk_body)
            if ch in seen_chunks:
                dup_chunks += 1
                continue
            seen_chunks.add(ch)
            when = page.get("crawled_at", "")
            out.append({
                "id": f"{p_idx}-{c_idx}",
                "url": page["url"],
                "title": title,
                "text": f"{title}\n{chunk_body}",
                "crawled_at": when,
                "fetched_at": when,
            })

    with CHUNKS_PATH.open("w", encoding="utf-8") as f:
        for row in out:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    lens = sorted(len(r["text"]) for r in out)
    print("עמודים שנקראו:       ", len(pages))
    print("עמודים ישנים הוסרו:  ", excluded)
    print("עמודים כפולים הוסרו: ", dup_pages)
    print("קטעים כפולים הוסרו:  ", dup_chunks)
    print("קטעים קצרים הוסרו:   ", short)
    print("קטעים שנשמרו:        ", len(out))
    print(f"אורך: מינימום {lens[0]} | חציון {int(statistics.median(lens))} | מקסימום {lens[-1]}")


if __name__ == "__main__":
    main()