
import sys

import rag

sys.stdout.reconfigure(encoding="utf-8")

# "in" = יש באתר, "out" = אין באתר, "personal" = פוליסה אישית
QUESTIONS = [
    ("יש לי נזילה בבית, איך אני מגישה תביעה?", "in"),
    ("מה עושים כשיש הצפה בדירה?", "in"),
    ("האם ביטוח דירה מכסה נזקי מים?", "in"),
    ("מה זה ביטוח תכולה?", "in"),
    ("מה כולל ביטוח מבנה?", "in"),
    ("האם הביטוח מכסה פריצה לדירה?", "in"),
    ("איך מגישים תביעה בביטוח רכב?", "in"),
    ("מה ההבדל בין ביטוח מקיף לביטוח צד ג'?", "in"),
    ("איך מכינים עוגת שוקולד?", "out"),
    ("מהי בירת צרפת?", "out"),
    ("איך מתקנים תקלה במחשב נייד?", "out"),
    ("האם הפוליסה שלי מכסה את הנזק?", "personal"),
]


def main():
    summary = []

    for question, label in QUESTIONS:
        hits = rag.retrieve(question, k=3)
        best = hits[0]["score"] if hits else 0.0

        print("=" * 70)
        print(f"[{label}] {question}")

        for i, hit in enumerate(hits, start=1):
            first_line = hit["text"].replace("\n", " ")[:110]
            print(f"  {i}. {hit['score']:.3f} | {hit['url']}")
            print(f"     {first_line}")

        summary.append((label, best, question))

    print("\n" + "=" * 70)
    print("סיכום: הציון הגבוה ביותר לכל שאלה")

    for label, best, question in summary:
        print(f"  {best:.3f}  [{label:8}] {question}")

    in_scores = [score for label, score, _ in summary if label == "in"]
    out_scores = [score for label, score, _ in summary if label == "out"]

    print(f"\nשאלות 'in':  מינימום {min(in_scores):.3f}")
    print(f"שאלות 'out': מקסימום {max(out_scores):.3f}")

    if min(in_scores) > max(out_scores):
        mid = (min(in_scores) + max(out_scores)) / 2
        print(f"יש הפרדה ברורה. סף ראשוני אפשרי: {mid:.3f}")
    else:
        print("אין הפרדה ברורה בין שאלות in ל-out.")


if __name__ == "__main__":
    main()
