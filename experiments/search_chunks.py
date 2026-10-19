import json
from pathlib import Path

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

EMBEDDINGS_PATH = Path("data/embeddings.jsonl")
MODEL_NAME = "intfloat/multilingual-e5-small"


def main():
    records = [
        json.loads(line)
        for line in EMBEDDINGS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    if not records:
        raise SystemExit("לא נמצאו וקטורים בקובץ")

    model = SentenceTransformer(MODEL_NAME)

    while True:
        question = input("\nהקלידי שאלה (או exit לסיום): ").strip()

        if question.lower() == "exit":
            break

        if not question:
            continue

        query_vector = model.encode(
            ["query: " + question],
            normalize_embeddings=True,
        )

        document_vectors = [
            record["embedding"] for record in records
        ]

        scores = cosine_similarity(
            query_vector,
            document_vectors,
        )[0]

        # מיון הקטעים לפי ציון ההתאמה, מהגבוה לנמוך
        ranked_indices = scores.argsort()[::-1]

        # בחירת קטעים מקישורים ומכותרות שונות
        top_indices = []
        seen_urls = set()
        seen_titles = set()

        for index in ranked_indices:
            record = records[index]
            url = record["url"]
            title = record["title"]

            if url in seen_urls or title in seen_titles:
                continue

            seen_urls.add(url)
            seen_titles.add(title)
            top_indices.append(index)

            if len(top_indices) == 5:
                break

        print("\nחמשת הקטעים המתאימים ביותר:\n")

        for rank, index in enumerate(top_indices, start=1):
            record = records[index]
            print(f"{rank}. ציון התאמה: {scores[index]:.4f}")
            print(f"כותרת: {record['title']}")
            print(f"קישור: {record['url']}")
            print(f"קטע: {record['text'][:500]}")
            print("-" * 60)


if __name__ == "__main__":
    main()
