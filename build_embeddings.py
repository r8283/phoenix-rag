
import json
from pathlib import Path

from sentence_transformers import SentenceTransformer

INPUT_PATH = Path("data/chunks.jsonl")
OUTPUT_PATH = Path("data/embeddings.jsonl")
MODEL_NAME = "intfloat/multilingual-e5-small"
BATCH_SIZE = 32

def main():
    if not INPUT_PATH.exists():
        raise SystemExit(f"קובץ הקלט לא נמצא: {INPUT_PATH}")

    chunks = [
        json.loads(line)
        for line in INPUT_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    if not chunks:
        raise SystemExit("לא נמצאו קטעים בקובץ הקלט")

    print("מספר קטעים:", len(chunks))
    print("טוען את המודל...")
    model = SentenceTransformer(MODEL_NAME)

    texts = [
        "passage: " + chunk["text"]
        for chunk in chunks
    ]

    print("יוצר וקטורים...")
    vectors = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    print("שומר את הווקטורים...")
    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        for chunk, vector in zip(chunks, vectors):
            record = {
                "id": chunk["id"],
                "url": chunk["url"],
                "title": chunk["title"],
                "text": chunk["text"],
                "fetched_at": chunk["fetched_at"],
                "embedding": vector.tolist(),
            }
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("הסתיים בהצלחה!")
    print("מספר וקטורים:", len(vectors))
    print("ממדים בכל וקטור:", len(vectors[0]))
    print("קובץ הפלט:", OUTPUT_PATH)
    print("גודל הקובץ בבתים:", OUTPUT_PATH.stat().st_size)


if __name__ == "__main__":
    main()
