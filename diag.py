"""
diag.py - מאבחן למה שאלה נענתה "לא מופיע באתר" למרות ציון אחזור גבוה.

עבור כל שאלה הוא מדפיס:
  1. את הקטעים שהאחזור מצא (ציון, עמוד, תחילת הטקסט)
  2. את ההחלטה של המודל הקל בשלוש גרסאות של הפרומפט:
     הנוכחי, בלי כלל 8 (בטיחות), בלי כללים 7 ו-8 (הכללים המחמירים)

שימוש:
  python diag.py
"""
import re
import sys

import rag

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

QUESTIONS = [
    "מה ההבדל בין ביטוח רכב מקיף לבין ביטוח צד ג'?",
    "איך מגישים תביעה בגין נכות מתאונה?",
]
MODELS = [("gemini-3.5-flash-lite", None)]

VARIANTS = [
    ("הפרומפט הנוכחי", []),
    ("בלי כלל 8 (בטיחות)", [8]),
    ("בלי כללים 7 ו-8", [7, 8]),
]


def drop_rules(template, numbers):
    for n in numbers:
        template = re.sub(rf"^{n}\. .*\n", "", template, flags=re.M)
    return template


for q in QUESTIONS:
    print("=" * 70)
    print("שאלה:", q)
    hits = rag.retrieve(q)
    for i, h in enumerate(hits, start=1):
        print(f"  [{i}] {h['score']:.3f} | {h['url'].split('.co.il')[-1]}")
        print("      " + h["text"].replace("\n", " ")[:220])

    excerpts = "\n\n".join(
        f"[{i}] ({h['title']})\n{h['text']}" for i, h in enumerate(hits, start=1)
    )
    for label, drop in VARIANTS:
        prompt = (
            drop_rules(rag.ANSWER_PROMPT, drop)
            .replace("{{EXCERPTS}}", excerpts)
            .replace("{{QUESTION}}", q)
        )
        text, model = rag.call_llm(
            prompt, json_mode=True, models=MODELS, validate=rag._valid_answer_json
        )
        data = rag.parse_json(text) if text else None
        if not data:
            print(f"\n  >> {label}: אין תשובה תקינה")
            continue
        print(f"\n  >> {label}: status={data.get('status')} used={data.get('used')}")
        print("     " + (data.get("answer") or "").replace("\n", " ")[:250])