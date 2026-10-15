"""
הגדרות מרכזיות לפרויקט RAG של אתר הפניקס.
יום 1 משתמש בעיקר ב-BASE_URL, DELAY, MAX_PAGES, INCLUDE/EXCLUDE ו-MIN_PAGE_TEXT_LEN.
שאר השדות (chunks, Chroma, Gemini) מוכנים לימים הבאים — בלי OpenAI.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
PAGES_PATH = DATA_DIR / "pages.jsonl"
SEED_URLS_PATH = DATA_DIR / "seed_urls.txt"
CHROMA_DIR = ROOT / "chroma_db"

# --- אתר ---
BASE_URL = "https://www.fnx.co.il"
ALLOWED_HOSTS = ("www.fnx.co.il", "fnx.co.il")

# --- סריקה (יום 1) ---
MAX_PAGES = 600
DELAY_SEC = 2.5
MIN_PAGE_TEXT_LEN = 300

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36 PhoenixRAG-Hackathon"# ברשתות עם NetFree/פרוקסי לפעמים צריך false; במחשב רגיל אפשר true
HTTP_SSL_VERIFY = os.getenv("HTTP_SSL_VERIFY", "false").strip().lower() in ("1", "true", "yes")

# אם INCLUDE ריק — כל כתובת מה-sitemap שעוברת EXCLUDE (מאות עמודים).
# אפשר למלא מילים כדי להצר לנושאים ספציפיים (תביעות, FAQ וכו').
URL_INCLUDE_KEYWORDS: tuple[str, ...] = ()

# כתובות שמכילות אחת מהמילים האלה — לא נסרקות
URL_EXCLUDE_KEYWORDS: tuple[str, ...] = (
    "/login",
    "/log-in",
    "/signin",
    "/signup",
    "/register",
    "/my-account",
    "/portal",
    "/search",
    "/tag/",
    "/author/",
    "/wp-admin",
    "/careers/",
    "/job/",
    "/media/",
    "/press-release/",
    ".pdf",
    ".jpg",
    ".png",
    ".gif",
    ".zip",
    "/api/",
    "/graphql",
    "javascript:",
    # מ-robots.txt של fnx.co.il
    "/testimtempp",
    "/info/default.aspx",
    "/spf",
    "/mobile-tzair",
    "/sales/pages/car/carinsurancenew.aspx",
    "/sales/pages/travel/travelinsurancenew.aspx",
    "/servicepages/sharedpages/discountsimulator",
    "/marketingcalculators",
    "/news/cooperation-with-neighbors.aspx",
    "?page=",
    "?domain=",
    "?tag=",
    "?staff=",
    "/more-information/",
    "/sites/docs/polarchive",
)

# --- חיתוך (יום 2+) ---
CHUNK_MIN_CHARS = 800
CHUNK_MAX_CHARS = 1200
CHUNK_OVERLAP_CHARS = 120

# --- מודלים (ימים 2–3; אין OpenAI) ---
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
GEMINI_MODEL = "gemini-3.8-flash"

# עדיפות 0 = הכי חשוב. כתובת מקבלת את הדרגה של התבנית הראשונה שמתאימה.
URL_PRIORITY_PATTERNS: tuple[tuple[int, tuple[str, ...]], ...] = (
    (0, ("claim", "apartment-insurance", "faq", "questions", "תביע")),
    (1, ("car-insurance", "health-insurance", "life-insurance", "business", "travel", "mortgage")),
    (2, ("insurance", "pension", "gemel", "service", "contact")),
)
URL_PRIORITY_DEFAULT = 3

# עמודים שאין בהם תוכן אמיתי
URL_EXCLUDE_KEYWORDS_EXTRA: tuple[str, ...] = ("/404", "agents-promotions")