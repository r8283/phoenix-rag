"""
eval.py - מריץ את questions.json על הבוט ומדפיס טבלה ותקציר.

שימוש:
  python eval.py                  # הרצה מלאה
  python eval.py --limit 3        # בדיקה מהירה על 3 השאלות הראשונות
  python eval.py --min-score 0    # בלי השער הראשון: כל שאלה מגיעה למודל (לכיול הסף)
  python eval.py --no-rewrite     # בלי ניסוח מחדש (להשוואה)

קורא:  questions.json
כותב:  eval_results.json  (נתונים מלאים)
       eval_answers.txt   (שאלות, תשובות ומקורות, לקריאה ידנית מול האתר)

סוגי שאלות:
  in        יש תשובה באתר. הבוט צריך לענות, עם מקור מהעמודים הצפויים.
  near      קרוב לתחום אבל אין באתר. הבוט צריך לסרב.
  out       מחוץ לתחום. הבוט צריך לסרב.
  personal  פוליסה אישית. הבוט צריך לסרב (עדיף עם הפניה לשירות הלקוחות).
"""
import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import rag

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

QUESTIONS_PATH = Path("questions.json")
RESULTS_PATH = Path("eval_results.json")
ANSWERS_PATH = Path("eval_answers.txt")
PAUSE_SEC = 1.0  # הפסקה בין שאלות, כדי לא לחרוג ממגבלת הקצב של ה-API

REFUSE_STATUSES = ("not_found", "personal_only")
BROKEN_STATUSES = ("llm_unavailable", "error")


def judge(q, res):
    status = res.get("status")
    sources = res.get("sources") or []
    if status in BROKEN_STATUSES:
        return "תקלה"
    if q["type"] == "in":
        if status != "answered":
            return "לא ענה"
        expect = q.get("expect_any")
        if expect and not any(part in url for url in sources for part in expect):
            return "מקור לא צפוי"
        return "עבר"
    # near / out / personal: הבוט צריך לסרב
    return "עבר" if status in REFUSE_STATUSES else "המציא"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--min-score", type=float, default=None)
    ap.add_argument("--no-rewrite", action="store_true")
    args = ap.parse_args()

    rag.USE_CACHE = False   # הבדיקה מודדת את החיפוש והמודל, לא את המטמון
    if args.min_score is not None:
        rag.MIN_SCORE = args.min_score
    if args.no_rewrite:
        rag.REWRITE = False

    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    if args.limit:
        questions = questions[: args.limit]
    print(f"מריץ {len(questions)} שאלות | MIN_SCORE={rag.MIN_SCORE} | REWRITE={rag.REWRITE}\n")

    rows, done = [], {}
    for q in questions:
        history = []
        prev = done.get(q.get("follow_up_of"))
        if prev:
            history = [
                {"role": "user", "content": prev["question"]},
                {"role": "assistant", "content": prev["answer"]},
            ]
        t0 = time.time()
        try:
            res = rag.answer(q["question"], history)
        except Exception as e:
            res = {"answer": f"שגיאה: {e}", "sources": [], "status": "error",
                   "top_score": 0.0, "model": None, "standalone": ""}
        secs = time.time() - t0

        row = {
            **q,
            "answer": res.get("answer", ""),
            "sources": res.get("sources", []),
            "status": res.get("status"),
            "top_score": float(res.get("top_score") or 0),
            "model": res.get("model"),
            "standalone": res.get("standalone", ""),
            "seconds": round(secs, 1),
            "verdict": judge(q, res),
        }
        rows.append(row)
        done[q["id"]] = row
        print(
            f"{q['id']:>2} {q['type']:8} {row['verdict']:12} {str(row['status']):15} "
            f"{row['top_score']:.3f} {secs:5.1f}s  {q['question'][:40]}"
        )
        time.sleep(PAUSE_SEC)

    # ---------- תקציר ----------
    ok = [r for r in rows if r["verdict"] != "תקלה"]
    broken = len(rows) - len(ok)
    ins = [r for r in ok if r["type"] == "in"]
    refs = [r for r in ok if r["type"] != "in"]
    in_pass = sum(r["verdict"] == "עבר" for r in ins)
    ref_pass = sum(r["verdict"] == "עבר" for r in refs)

    print("\n" + "=" * 64)
    print(f"שאלות שיש להן תשובה באתר: {in_pass}/{len(ins)} עברו")
    for r in ins:
        if r["verdict"] != "עבר":
            print(f"   {r['verdict']}: [{r['id']}] {r['question']} (ציון {r['top_score']:.3f})")

    print(f"שאלות שצריך לסרב להן:     {ref_pass}/{len(refs)} עברו   <- חצי מהציון")
    for r in refs:
        if r["verdict"] == "המציא":
            print(f"   המציא: [{r['id']}] {r['question']} (ציון {r['top_score']:.3f})")
    personal = [r for r in refs if r["type"] == "personal"]
    if personal:
        n = sum(r["status"] == "personal_only" for r in personal)
        print(f"   מתוכן שאלות אישיות שקיבלו הפניה לשירות הלקוחות: {n}/{len(personal)}")

    print(f"תקלות (מודל לא זמין):     {broken}  (לא נספרות בציון. כדאי להריץ שוב)")
    if rows:
        times = [r["seconds"] for r in rows]
        print(f"זמן ממוצע: {sum(times) / len(times):.1f} שנ' | הכי איטי: {max(times):.1f} שנ'")
    print("מודלים ששימשו:", dict(Counter(r["model"] for r in rows if r["model"])))

    # עזר לכיול הסף
    in_scores = [r["top_score"] for r in ins]
    ref_scores = [r["top_score"] for r in refs if r["type"] in ("out", "near")]
    if in_scores and ref_scores:
        lo, hi = min(in_scores), max(ref_scores)
        print(f"\nכיול: הציון הנמוך ביותר בשאלות 'in' = {lo:.3f} | הגבוה ביותר ב-'out/near' = {hi:.3f}")
        if lo > hi:
            print(f"   יש הפרדה. סף בטוח ל-MIN_SCORE: בין {hi:.3f} ל-{lo:.3f}")
        else:
            print("   יש חפיפה בין הציונים, ולכן הסף לבדו לא מספיק. ההכרעה נשארת למודל השפה.")

    # ---------- שמירה ----------
    RESULTS_PATH.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    with ANSWERS_PATH.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(f"[{r['id']}] ({r['type']}) {r['verdict']} | סטטוס: {r['status']} | ציון: {r['top_score']:.3f}\n")
            f.write(f"שאלה: {r['question']}\n")
            f.write(f"שאלה עצמאית: {r['standalone']}\n")
            f.write(f"תשובה: {r['answer']}\n")
            for u in r["sources"]:
                f.write(f"מקור: {u}\n")
            f.write("-" * 64 + "\n")
    print(f"\nנשמר: {RESULTS_PATH} ו-{ANSWERS_PATH}")


if __name__ == "__main__":
    main()