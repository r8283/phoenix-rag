import rag

QS = [
    "מה כולל ביטוח מבנה?",
    "כמה שילמתי השנה על הביטוח שלי?",
    "האם כדאי לי להגדיל את הכיסוי בפוליסה שלי?",
]
for q in QS:
    r = rag.answer(q, [])
    print("\n", q, "|", r["status"])
    print(r["answer"])
    for s in r["sources"]:
        print("   ", s)