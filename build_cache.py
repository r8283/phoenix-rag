"""
build_cache.py - בונה מטמון של תשובות שנבדקו מראש לשאלות הדמו.

קורא:  questions.json
כותב:  data/answer_cache.json

כל שאלה רצה דרך rag.answer (בלי מטמון), אחת אחרי השנייה עם הפסקה, כדי לא לחרוג ממכסת Gemini.
נשמרות רק תשובות שעברו את הבדיקה של eval.py (judge) ושהסטטוס שלהן answered, not_found,
personal_only או meta. תשובת "השירות עמוס" או שגיאה לא נשמרות, ומקבלות עוד ניסיון אחד בסוף.
שאלת המשך נשמרת יחד עם השאלה שלפניה, ונשלפת רק כשגם השאלה הקודמת בשיחה זהה.

שימוש:  python build_cache.py              # כל השאלות
        python build_cache.py --only 8 20  # רק שאלות 8 ו-20. שאר המטמון נשאר כמו שהוא
"""
import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import rag
from eval import judge

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

QUESTIONS_PATH = Path("questions.json")
PAUSE_SEC = 3.0


def ask(q, done):
    history = []
    prev = done.get(q.get("follow_up_of"))
    if prev:
        history = [{"role": "user", "content": prev["question"]},
                   {"role": "assistant", "content": prev["answer"]}]
    t = time.time()
    try:
        res = rag.answer(q["question"], history)
    except Exception as e:
        res = {"status": "error", "answer": str(e), "sources": []}
    return res, history, time.time() - t


def entry(q, res, history):
    prev = rag._prev_question(history)
    return {
        "key": rag.cache_key(q["question"]),
        "prev_key": rag.cache_key(prev) if prev else None,
        "question": q["question"],
        "prev_question": prev,
        "status": res.get("status"),
        "answer": res.get("answer", ""),
        "sources": res.get("sources") or [],
        "followups": res.get("followups") or [],
        "model": res.get("model"),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=int, nargs="+", help="מספרי השאלות לבנייה מחדש")
    args = ap.parse_args()

    rag.USE_CACHE = False   # תמיד תשובות חדשות
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    done, saved, failed = {}, {}, []
    if args.only:
        # שאר התשובות נשארות. הן משמשות גם כהקשר לשאלות המשך שנבנות מחדש
        old = {e["question"]: e for e in json.loads(rag.CACHE_PATH.read_text(encoding="utf-8"))}
        for q in questions:
            if q["question"] in old:
                saved[q["id"]] = old[q["question"]]
                done[q["id"]] = {"question": q["question"], "answer": old[q["question"]]["answer"]}
        questions_all, questions = questions, [q for q in questions if q["id"] in args.only]
        for q in questions:
            saved.pop(q["id"], None)
    else:
        questions_all = questions

    def run(q):
        if q.get("follow_up_of") and q["follow_up_of"] not in done:
            print(f"{q['id']:>2} דילוג: השאלה הקודמת ({q['follow_up_of']}) לא נשמרה", flush=True)
            return False   # בלי ההקשר היא הייתה נשמרת כשאלה עצמאית
        res, history, sec = ask(q, done)
        verdict = judge(q, res)
        ok = res.get("status") in rag.CACHE_STATUSES and verdict == "עבר"
        print(f"{q['id']:>2} {res.get('status')!s:14} {sec:5.1f}s {verdict:12} {q['question']}", flush=True)
        if ok:
            done[q["id"]] = {"question": q["question"], "answer": res.get("answer", "")}
            saved[q["id"]] = entry(q, res, history)
        return ok

    for q in questions:
        if not run(q):
            failed.append(q)
        time.sleep(PAUSE_SEC)

    # עוד ניסיון אחד לשאלות שנכשלו (בדרך כלל עומס על המודל)
    still = []
    for q in failed:
        if not run(q):
            still.append(q)
        time.sleep(PAUSE_SEC)

    entries = [saved[q["id"]] for q in questions_all if q["id"] in saved]
    rag.CACHE_PATH.write_text(json.dumps(entries, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nבמטמון {len(entries)}/{len(questions_all)} תשובות ב-{rag.CACHE_PATH}")
    if still:
        print("לא נשמרו:", ", ".join(f"{q['id']} ({q['question']})" for q in still))


if __name__ == "__main__":
    main()
