import json
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

sys.stdout.reconfigure(encoding="utf-8")

EMBEDDINGS_PATH = Path("data/embeddings.jsonl")
MODEL_NAME = "intfloat/multilingual-e5-small"
TOP_K = 3

# "in" = יש באתר, "out" = אין באתר, "personal" = פוליסה אישית
QUESTIONS = [
    ("יש לי נזילה בבית, איך אני מגישה תביעה?", "in"),
    ("מה עושים כשיש הצפה בדירה?", "in"),
    ("האם ביטוח דירה מכסה נזקי מים?", "in"),
    ("מה זה ביטוח תכולה?", "in"),
    ("מה כולל ביטוח מבנה?", "in"),
    ("האם הביטוח מכסה פריצה לדירה?", "in"),
    ("איך מגישים תביעה בביטוח רכב?", "in"),
    ("מה ההבדל בין ביטוח מקיף לביטוח צד ג?", "in"),
    ("איך מכינים עוגת שוקולד?", "out"),
    ("מהי בירת צרפת?", "out"),
    ("איך מתקנים תקלה במחשב נייד?", "out"),
    ("האם הפוליסה שלי מכסה את הנזק?", "personal"),
]


def main():
    records = [
        json.loads(line)
        for line in EMBEDDINGS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    matrix = np.array([r["embedding"] for r in records], dtype="float32")
    model = SentenceTransformer(MODEL_NAME)

    summary = []
    for question, label in QUESTIONS:
        q = model.encode(["query: " + question], normalize_embeddings=True)[0]
        scores = matrix @ q
        order = scores.argsort()[::-1]

        print("=" * 70)
        print(f"[{label}] {question}")

        shown, seen_text = 0, set()
        for idx in order:
            rec = records[idx]
            key = rec["text"][:200]
            if key in seen_text:
                continue
            seen_text.add(key)
            shown += 1
            first_line = rec["text"].replace("\n", " ")[:110]
            print(f"  {shown}. {scores[idx]:.3f} | {rec['url']}")
            print(f"     {first_line}")
            if shown == TOP_K:
                break
        summary.append((label, float(scores[order[0]]), question))

    print("\n" + "=" * 70)
    print("סיכום: הציון הטוב ביותר לכל שאלה")
    for label, best, question in summary:
        print(f"  {best:.3f}  [{label:8}] {question}")

    in_scores = [s for lbl, s, _ in summary if lbl == "in"]
    out_scores = [s for lbl, s, _ in summary if lbl == "out"]
    print(f"\nשאלות 'in':  מינימום {min(in_scores):.3f}")
    print(f"שאלות 'out': מקסימום {max(out_scores):.3f}")
    if min(in_scores) > max(out_scores):
        mid = (min(in_scores) + max(out_scores)) / 2
        print(f"יש הפרדה ברורה. סף ראשוני מוצע: כ-{mid:.3f}")
    else:
        print("אין הפרדה ברורה בין in ל-out.")


if __name__ == "__main__":
    main()