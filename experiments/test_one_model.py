import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
key = os.getenv("GEMINI_API_KEY")

if not key:
    raise SystemExit("לא נמצא מפתח בקובץ .env")

client = genai.Client(api_key=key)

try:
    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents="Reply with exactly: עובד",
    )
    print("הצלחה:", response.text)

except Exception as e:
    print("סוג השגיאה:", type(e).__name__)
    print("פרטי השגיאה:", str(e))