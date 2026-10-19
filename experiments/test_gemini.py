import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
key = os.getenv("GEMINI_API_KEY")
if not key:
    raise SystemExit("לא נמצא GEMINI_API_KEY בקובץ .env")

client = genai.Client(api_key=key)

CANDIDATES = [
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-3-flash-preview",
    "gemma-4-31b-it",
    "gemma-4-26b-a4b-it",
    "gemini-flash-latest",
    "gemini-flash-lite-latest",
]

working = []
for model in CANDIDATES:
    try:
        r = client.models.generate_content(
            model=model,
            contents="כתוב משפט אחד בעברית: מה זה ביטוח דירה?",
        )
        print(f"עובד    {model}: {r.text.strip()[:70]}")
        working.append(model)
    except Exception as e:
        code = str(e)[:3]
        label = {"503": "עמוס  ", "404": "לא קיים", "429": "מכסה  "}.get(code, "שגיאה ")
        print(f"{label} {model}")

print("\nעובדים:", working)