import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from google import genai
from google.genai import types

import rag

sys.stdout.reconfigure(encoding="utf-8")
load_dotenv()

TIMEOUT_MS = 60000
client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY"),
    http_options=types.HttpOptions(timeout=TIMEOUT_MS),
)

QUESTION = "איך מגישים תביעת ביטוח בגין נזק מים בדירה?"

print("מכין את הקטעים...")
hits = rag.retrieve(QUESTION)
excerpts = "\n\n".join(
    f"[{i}] ({h['title']})\n{h['text']}" for i, h in enumerate(hits, start=1)
)
prompt = (
    rag.ANSWER_PROMPT
    .replace("{{EXCERPTS}}", excerpts)
    .replace("{{QUESTION}}", QUESTION)
)
print(f"שאלה: {QUESTION}")
print(f"ציון הקטע הטוב ביותר: {hits[0]['score']:.3f} | אורך הפרומפט: {len(prompt)} תווים\n")

VARIANTS = [
    ("gemini-3.8-flash", None),
    ("gemini-3.8-flash", "low"),
    ("gemini-3.7-flash", None),
    ("gemini-3.7-flash", "low"),
    ("gemini-3.1-flash-lite", None),
    ("gemini-3.5-flash-lite", None),
    ("gemma-4-31b-it", None),
]


def run(model, thinking):
    label = f"{model}" + (f" (חשיבה {thinking})" if thinking else "")
    kwargs = {"temperature": 0}
    if not model.startswith("gemma"):
        kwargs["response_mime_type"] = "application/json"
    t0 = time.time()
    try:
        if thinking:
            kwargs["thinking_config"] = types.ThinkingConfig(thinking_level=thinking)
        cfg = types.GenerateContentConfig(**kwargs)
        r = client.models.generate_content(model=model, contents=prompt, config=cfg)
        secs = time.time() - t0
        data = rag.parse_json(r.text or "")
        if not data:
            return label, secs, "JSON לא תקין או ריק"
        used = data.get("used")
        body = (data.get("answer") or "").replace("\n", " ")[:70]
        return label, secs, f"status={data.get('status')} used={used} | {body}"
    except Exception as e:
        return label, time.time() - t0, "שגיאה: " + str(e)[:90]


print("שולח את כל הגרסאות במקביל (עד דקה)...\n")
results = []
with ThreadPoolExecutor(max_workers=len(VARIANTS)) as pool:
    futures = [pool.submit(run, m, t) for m, t in VARIANTS]
    for fut in as_completed(futures):
        label, secs, summary = fut.result()
        results.append((secs, label, summary))
        print(f"{secs:5.1f} שנ' | {label}\n        {summary}\n")

print("=" * 60)
print("מהמהיר לאיטי:")
for secs, label, summary in sorted(results):
    ok = "תקין" if summary.startswith("status=") else "בעיה"
    print(f"  {secs:5.1f} שנ' | {ok:4} | {label}")