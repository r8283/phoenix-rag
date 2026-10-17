import rag

H = [
    {"role": "user", "content": "מה זה ביטוח תכולה?"},
    {"role": "assistant", "content": "ביטוח תכולת דירה נועד לספק הגנה מפני נזקים שעלולים להיגרם לרכוש הנמצא בדירה, כגון רהיטים ומכשירי חשמל, מפני אירועים כמו שריפה, הצפה ופריצה."},
]

# שאלות לא קשורות אחרי שיחה על ביטוח: צפוי not_found
UNRELATED = [
    "יש מתכון לעוגה?",
    "איך מכינים עוגת שוקולד?",
    "מה מזג האוויר מחר?",
    "מהי בירת צרפת?",
    "איך מתקנים תקלה במחשב נייד?",
]
# שאלות המשך אמיתיות: צפוי answered
FOLLOW_UPS = [
    "ומה לא מכוסה בו?",
    "אפשר להוסיף הרחבות?",
]

bad = 0
for expected, qs in (("not_found", UNRELATED), ("answered", FOLLOW_UPS)):
    for q in qs:
        r = rag.answer(q, H)
        ok = r["status"] == expected
        bad += not ok
        print(("עבר " if ok else "נכשל"), r["status"], f"{r['top_score']:.3f}", "|", q, "->", r["standalone"])
print("\nכשלים:", bad)