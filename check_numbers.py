"""
check_numbers.py - בודק שכל מספר שמופיע בתשובות הבוט מופיע גם בעמודים שצוטטו.

מספרים הם המקום המסוכן ביותר להמצאה בביטוח: טלפונים, ימים, אחוזים, סכומים.
הבדיקה לא קוראת ל-API. היא עובדת על eval_results.json (שנוצר ב-eval.py)
ועל data/pages.jsonl.

שימוש:
  python check_numbers.py
"""
import json
import re
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

NUM = re.compile(r"\d[\d\-,.]*\d|\d")


def norm(token):
    """054-9893455 ו-0549893455 נחשבים זהים."""
    return re.sub(r"[\-,.]", "", token)


def numbers(text):
    return {norm(t) for t in NUM.findall(text)}


def main():
    pages = {}
    for line in Path("data/pages.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            pages[r["url"]] = r["text"]

    rows = json.loads(Path("eval_results.json").read_text(encoding="utf-8"))
    checked = suspicious = 0
    for r in rows:
        if r["status"] != "answered":
            continue
        checked += 1
        source_numbers = numbers(" ".join(pages.get(u, "") for u in r["sources"]))
        missing = sorted(n for n in numbers(r["answer"]) if n not in source_numbers)
        if missing:
            suspicious += 1
            print(f"[{r['id']}] מספרים שלא נמצאו במקורות שצוטטו: {missing}")
            print(f"     שאלה: {r['question']}")
            print(f"     תשובה: {r['answer'][:200]}")
    print(f"\nנבדקו {checked} תשובות. חשודות: {suspicious}")
    if suspicious:
        print("לכל מספר חשוד: לחפש אותו באתר. אם הוא מופיע בעמוד אחר, זו בעיית ייחוס מקור. אם לא, זו המצאה.")


if __name__ == "__main__":
    main()