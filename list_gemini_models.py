import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
key = os.getenv("GEMINI_API_KEY")

if not key:
    raise SystemExit("לא נמצא GEMINI_API_KEY בקובץ .env")

client = genai.Client(api_key=key)

try:
    print("מודלים שתומכים ביצירת תוכן:\n")

    for model in client.models.list():
        actions = getattr(model, "supported_actions", []) or []

        if "generateContent" in actions:
            print(model.name)

except Exception as e:
    print("סוג השגיאה:", type(e).__name__)
    print("פרטי השגיאה:", str(e))