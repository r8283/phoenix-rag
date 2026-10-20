"""
app.py - ממשק הצ'אט (Streamlit), בעיצוב של אפליקציית הודעות בטלפון.
הרצה:  python -m streamlit run app.py
לוגו: assets/logo.png (אם הקובץ קיים הוא מוצג בראש הצ'אט)
"""
import base64
import hashlib
import html
import json
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import streamlit as st
import streamlit.components.v1 as components

import rag


@st.cache_resource
def _warm_up():
    rag._load_index()
    return True


_warm_up()

LOGO = Path("assets/logo.png")
HAS_LOGO = LOGO.exists()
# לוגו מוטמע (גיבוי, אם הקובץ assets/logo.png לא קיים)
LOGO_B64 = "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"
PAGES_PATH = Path("data/pages.jsonl")

APP_TITLE = "עוזר דיגיטלי"
APP_SUBTITLE = "שאלות על שירותי הפניקס, תשובות מהאתר"
DISCLAIMER = (
    "פרויקט האקתון. השירות אינו אתר רשמי של הפניקס, אינו ייעוץ ביטוחי "
    "ואינו מחליף פנייה לשירות הלקוחות."
)
SITE_URL = "https://www.fnx.co.il"

# הצעות לשאלות המשך (כולן שאלות שנבדקו ויש להן תשובה באתר)
FOLLOWUPS_APARTMENT = [
    "מה זה ביטוח תכולה?",
    "מה לא מכוסה בביטוח תכולה?",
    "אילו הרחבות אפשר להוסיף לביטוח תכולת דירה?",
    "מה כולל ביטוח מבנה?",
    "האם ביטוח דירה מכסה נזקי מים?",
    "האם הביטוח מכסה פריצה לדירה?",
    "מה עושים אחרי פריצה לבית?",
    "אילו מסמכים צריך לצרף לתביעה בביטוח דירה?",
    "מה עושים כשיש הצפה בדירה?",
    "איך מתכוננים לחורף כדי למנוע נזקי רטיבות בבית?",
]
FOLLOWUPS_CAR = [
    "איך מגישים תביעה בביטוח רכב?",
    "מה ההבדל בין ביטוח מקיף לביטוח צד ג?",
    "איך מגישים תביעת רכב חובה בגין תאונת דרכים?",
]
FOLLOWUPS_OTHER = ["איך מגישים תביעה בגין נכות מתאונה?"]
CAR_WORDS = re.compile(r"רכב|מקיף|צד ג|חובה|תאונת דרכים")
HOME_WORDS = re.compile(r"דירה|בית|תכולה|מבנה|פריצה|נזילה|הצפה|רטיבות|חורף")

GREETING = (
    "שלום 👋 אני עוזר דיגיטלי שעונה על שאלות על ביטוח, פנסיה, חיסכון והלוואות, "
    "לפי תוכן אתר הפניקס. בכל תשובה אצרף קישור לעמוד המקור."
)
BUSY_MSG = "השירות עמוס כרגע. אפשר לנסות שוב בעוד רגע."
NOT_FOUND_HINT = ""

st.set_page_config(
    page_title=APP_TITLE,
    page_icon=str(LOGO) if HAS_LOGO else "💬",
    layout="centered",
    initial_sidebar_state="collapsed",
)


def logo_data_uri():
    if HAS_LOGO:
        try:
            return "data:image/png;base64," + base64.b64encode(LOGO.read_bytes()).decode()
        except Exception:
            pass
    return "data:image/png;base64," + LOGO_B64


# ---------- עיצוב (אפשר לשנות את הצבעים כאן) ----------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Assistant:wght@400;500;600;700;800&display=swap');
:root {
  --brand: #E4572E; --brand-dark: #C4411B; --brand-soft: #FDEEE8;
  --navy: #2B2D7F; --ink: #1F2933; --muted: #7B8794; --line: #ECE6E0; --wall: #F4EFEA;
}
html, body, .stApp, .stMarkdown, button, input, textarea {
  direction: rtl; font-family: 'Assistant', sans-serif !important;
}
.stApp { background: radial-gradient(1200px 500px at 50% -100px, #FFE3D6 0%, #F3F0EC 60%) fixed; }
[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stAppDeployButton"],
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"], #MainMenu, footer { display: none !important; }

/* מסגרת בצורת טלפון: על כל גובה המסך, והגלילה בתוכה */
[data-testid="stMain"], [data-testid="stAppScrollToBottomContainer"], section.main { overflow: hidden !important; }
.block-container {
  max-width: 430px !important; height: calc(100dvh - 2rem); margin: 1rem auto !important;
  padding: 0 0 150px 0 !important; overflow-y: auto; overflow-x: hidden;
  background-color: var(--wall);
  background-image: radial-gradient(#E6DCD2 1.2px, transparent 1.2px); background-size: 20px 20px;
  border-radius: 38px;
  box-shadow: 0 24px 60px rgba(43,45,127,.28), 0 0 0 10px #17181F, 0 0 0 11px #3B3D4A;
  scrollbar-width: none;
}
.block-container::-webkit-scrollbar { display: none; }
@media (max-width: 480px) {
  .block-container { height: 100dvh; margin: 0 auto !important; border-radius: 0; box-shadow: none; }
}
.block-container > div > div { padding-inline: 14px; }

/* כותרת עליונה: שורת מצב של טלפון (עם חור מצלמה) ומתחתיה כותרת האתר */
.phonetop { position: relative; margin: 0 -14px 14px -14px; box-shadow: 0 4px 14px rgba(43,45,127,.08);
  background: linear-gradient(180deg, #fff 0%, #FFF6F1 100%); border-bottom: 3px solid var(--brand); }
.statusbar {
  direction: ltr; position: absolute; top: 0; left: 0; right: 0; z-index: 2; height: 30px; padding: 0 22px;
  background: transparent; pointer-events: none;
  display: flex; align-items: center; justify-content: space-between;
  font-size: .78rem; font-weight: 700; color: #1F2933; letter-spacing: .2px;
}
.statusbar .cam {
  position: absolute; left: 50%; top: 9px; transform: translateX(-50%);
  width: 13px; height: 13px; border-radius: 50%;
  background: radial-gradient(circle at 35% 35%, #3a3f5c 0 18%, #0b0c12 45%);
  box-shadow: 0 0 0 2px #14151c;
}
.statusbar .icons { display: flex; align-items: center; gap: 5px; }
.statusbar svg { display: block; }
@media (max-width: 480px) { .statusbar { display: none; } .topbar { padding: 12px 14px !important; gap: 10px; } .topbar .t2 { font-size: .76rem; } }
.topbar {
  display: flex; align-items: center; gap: 14px; padding: 38px 18px 14px; margin: 0;
  background: transparent;
}
/* הכותרת קבועה לראש המסגרת (לא תלויה במיקום בתוך הדף), והתוכן גולל מתחתיה */
[data-testid="stElementContainer"]:has(.phonetop) {
  position: fixed !important; top: 1rem; left: 50%; transform: translateX(-50%);
  width: min(430px, 100vw) !important; z-index: 50; margin: 0 !important; padding: 0 !important;
}
.phonetop { margin: 0 !important; border-radius: 38px 38px 0 0; }
.block-container { padding-top: 98px !important; }
@media (max-width: 480px) {
  [data-testid="stElementContainer"]:has(.phonetop) { top: 0; }
  .phonetop { border-radius: 0; }
  .block-container { padding-top: 76px !important; }
}
.topbar img { height: 38px; width: auto; display: block; mix-blend-mode: multiply; }
.topbar .sep { width: 1px; height: 34px; background: #E7DCD3; }
.topbar .txt { flex: 1; text-align: center; }
.topbar .t1 { font-weight: 700; color: var(--navy); font-size: 1.35rem; line-height: 1.2; letter-spacing: .2px; }
.topbar .t2 { color: #6B7280; font-size: .86rem; font-weight: 400; margin-top: 3px; }


/* הצעות לשאלות נוספות */
.followups-title { font-size: .9rem; font-weight: 600; color: #4A5568; margin: 10px 4px 6px; }
.st-key-followups { background: #fff; border-radius: 16px; padding: 10px 12px 12px; margin-top: 6px;
  box-shadow: 0 1px 6px rgba(0,0,0,.06); }
.st-key-followups [data-testid="stElementContainer"] { width: 100%; }
.st-key-followups .stButton { display: flex; justify-content: flex-start; }
.st-key-followups .stButton > button {
  width: auto; max-width: 100%; border-radius: 999px; padding: 6px 16px; min-height: 0;
  background: #fff; border: 1px solid #C9CDF0; color: var(--navy); font-weight: 500;
}
.st-key-followups .stButton > button:hover { background: #F2F3FF; border-color: var(--navy); color: var(--navy); }
.st-key-followups .stButton > button p { text-align: right; }

/* בועות */
.row { display: flex; margin: 6px 0; }
.row.bot { justify-content: flex-start; }   /* בעברית ההתחלה היא מימין */
.row.user { justify-content: flex-end; }
.bubble {
  max-width: 86%; padding: 10px 14px 8px 14px; font-size: .98rem; line-height: 1.7;
  box-shadow: 0 1px 2px rgba(0,0,0,.08); word-wrap: break-word; color: var(--ink);
}
.row.user .bubble {
  background: linear-gradient(135deg, #E4572E, #EE7A4E); color: #fff;
  border-radius: 18px 18px 18px 4px;
}
.row.bot .bubble { background: #fff; border-radius: 18px 18px 4px 18px; text-align: right; direction: rtl; }
.row.bot .bubble.info { background: #F1F5F9; }
.row.bot .bubble.warn { background: #FFF6DD; color: #6B4E00; }
.time { font-size: .68rem; opacity: .6; margin-top: 2px; text-align: end; }
.hint { color: var(--muted); font-size: .85rem; margin-top: 6px; }

/* מקורות */
.sources-title { color: var(--muted); font-size: .78rem; margin: 10px 0 5px 0; font-weight: 500; }
.chips { display: flex; flex-wrap: wrap; gap: 6px; }
a.chip {
  display: inline-flex; align-items: center; gap: 6px; padding: 4px 10px 4px 8px; border-radius: 999px;
  background: var(--brand-soft); border: 1px solid #F6CDBD; font-size: .8rem; text-decoration: none !important;
  color: var(--brand-dark) !important; line-height: 1.4;
}
a.chip:hover { background: var(--brand); color: #fff !important; }
a.chip .n { background: #fff; border-radius: 50%; width: 18px; height: 18px; font-size: .7rem; font-weight: 700;
  display: inline-flex; align-items: center; justify-content: center; flex-shrink: 0; }

/* מצב "מקליד" */
.typing { display: inline-flex; gap: 5px; padding: 6px 2px; }
.typing span { width: 8px; height: 8px; border-radius: 50%; background: #B8B2AB; animation: blink 1.2s infinite; }
.typing span:nth-child(2) { animation-delay: .2s; } .typing span:nth-child(3) { animation-delay: .4s; }
@keyframes blink { 0%,60%,100% { opacity: .25; transform: translateY(0);} 30% { opacity: 1; transform: translateY(-3px);} }

/* כפתורים */
.stButton > button {
  border-radius: 999px; border: 1px solid #F1D3C6; background: #fff; color: var(--ink);
  padding: .5rem 1rem; font-size: .92rem; box-shadow: 0 1px 2px rgba(0,0,0,.05);
}
.stButton > button:hover { background: var(--brand-soft); border-color: var(--brand); color: var(--brand-dark); }
.stButton > button p { text-align: right; }

/* כפתור שאלה בקול: אייקון בלי עיגול, בתוך תיבת הכתיבה, מימין לכפתור השליחה */
[data-testid="stElementContainer"]:has([data-testid="stPopover"]), [data-testid="stLayoutWrapper"]:has(> [data-testid="stPopover"]) {
  position: fixed !important; bottom: 44px; left: calc(50% - 145px);
  z-index: 300; width: 32px !important; min-width: 0 !important; margin: 0 !important; padding: 0 !important; gap: 0 !important;
}
[data-testid="stPopover"] { width: 32px; }
[data-testid="stPopover"] button {
  width: 32px; height: 32px; min-height: 32px; border-radius: 50%; padding: 0 !important;
  background: transparent !important; border: none !important; box-shadow: none !important; color: var(--brand);
  display: flex !important; align-items: center; justify-content: center;
}
[data-testid="stPopover"] button:hover { background: var(--brand-soft) !important; transform: none; }
[data-testid="stPopover"] button div[aria-hidden="true"], [data-testid="stPopover"] button [data-testid="stMarkdownContainer"] { display: none !important; }
[data-testid="stPopover"] button > div, [data-testid="stPopover"] button > div > div, [data-testid="stPopover"] button span {
  display: flex; align-items: center; justify-content: center; margin: 0 !important; padding: 0 !important; width: auto; gap: 0 !important;
}
[data-testid="stPopover"] button [data-testid="stIconMaterial"] { font-size: 1.5rem; color: var(--brand); line-height: 1; }
[data-testid="stPopoverBody"] { min-width: 300px; direction: rtl; }
/* חלון ההקלטה: בתוך המסגרת ומעל תיבת הכתיבה (ברירת המחדל ממקמת אותו לפי הכפתור, ובמחשב הוא יצא מהמסגרת) */
[data-testid="stPopoverBody"] {
  transform: none !important; top: auto !important; left: calc(50% - 201px) !important;
  bottom: 98px !important; width: 402px !important; max-width: 402px !important;
}
@media (max-width: 480px) {
  [data-testid="stElementContainer"]:has([data-testid="stPopover"]), [data-testid="stLayoutWrapper"]:has(> [data-testid="stPopover"]) { left: 84px; bottom: 26px; }
  [data-testid="stPopoverBody"] { left: 8px !important; right: 8px !important; width: auto !important; max-width: none !important; bottom: 80px !important; }
}

/* קלט הודעה: פס אטום ברוחב המסגרת, בתחתית שלה. תוכן שנגלל נבלע מאחוריו */
[data-testid="stBottom"] { position: fixed !important; bottom: 1rem !important; left: 0; right: 0; width: 100%; z-index: 100; }
[data-testid="stBottom"], [data-testid="stBottom"] > div { background: transparent !important; }
[data-testid="stBottomBlockContainer"] {
  max-width: 430px !important; margin: 0 auto; padding-inline: 14px !important;
  padding-top: 12px !important; padding-bottom: 14px !important;
  background: #F4EFEA !important; border-radius: 0 0 38px 38px;
  box-shadow: 0 -8px 14px -8px rgba(0,0,0,.10);
}
@media (max-width: 480px) {
  [data-testid="stBottom"] { bottom: 0 !important; }
  [data-testid="stBottomBlockContainer"] { border-radius: 0; padding-bottom: 12px !important; }
}
[data-testid="stChatInput"] {
  background: #fff; border-radius: 999px; border: 1px solid #E5DED7; box-shadow: 0 2px 10px rgba(0,0,0,.10);
}
[data-testid="stChatInput"] textarea { font-size: 1rem; padding-left: 44px !important; }
/* בזמן שהבוט חושב: התיבה נעולה, וזה צריך להיראות בבירור (המסך עצמו כבר לא מתעמעם) */
[data-testid="stChatInput"]:has(textarea:disabled) { opacity: .5; }
[data-testid="stChatInput"] textarea:disabled { cursor: not-allowed; }

/* בלי הבהוב בזמן run: Streamlit מעמעם אלמנטים "ישנים" ומציג אינדיקטור ריצה */
[data-stale="true"], [data-stale="true"] * { opacity: 1 !important; transition: none !important; }
[data-testid="stStatusWidget"], [data-testid="stAppRunningIcon"], [data-testid="stSpinner"] { display: none !important; }
.stApp, .stApp > header, [data-testid="stAppViewContainer"], [data-testid="stMain"] { transition: none !important; }
[data-testid="stChatInputSubmitButton"] { background: var(--brand) !important; color: #fff !important; border-radius: 50%; }

[data-testid="stExpander"] { background: #fff; border-radius: 14px; border: 1px solid var(--line); }
[data-testid="stExpander"] summary { justify-content: center; text-align: center; }
[data-testid="stExpander"] summary { position: relative; }
[data-testid="stExpander"] summary > span { width: 100%; justify-content: center; }
[data-testid="stExpander"] summary > span > span:first-child { position: absolute; left: 12px; top: 50%; transform: translateY(-50%); }
[data-testid="stExpander"] summary [data-testid="stMarkdownContainer"] { text-align: center !important; width: 100%; }
[data-testid="stExpander"] summary p { text-align: center; }
[data-testid="stExpander"] [data-testid="stExpanderDetails"],
[data-testid="stExpander"] [data-testid="stExpanderDetails"] * { text-align: right !important; direction: rtl; }
.disclaimer { color: var(--muted); font-size: .7rem; text-align: center; margin: 16px 6px 0 6px; line-height: 1.5; }
a { color: var(--brand-dark); }
</style>
""",
    unsafe_allow_html=True,
)


# ---------- כותרות עמודים (לתצוגת המקורות) ----------
@st.cache_data(show_spinner=False)
def load_titles():
    """url -> כותרת העמוד, מתוך data/pages.jsonl. אם אין, משתמשים בחלק מהכתובת."""
    titles = {}
    if not PAGES_PATH.exists():
        return titles
    try:
        with open(PAGES_PATH, encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                t = (d.get("title") or d.get("page_title") or "").strip()
                if d.get("url") and t:
                    titles[d["url"]] = t
    except Exception:
        pass
    return titles


def source_label(url):
    t = load_titles().get(url)
    if t:
        for sep in (" | ", " - ", " – "):
            if sep in t:
                t = t.split(sep)[0]
        return t if len(t) <= 45 else t[:42] + "..."
    slug = url.rstrip("/").split("/")[-1]
    return slug.replace("-", " ").replace("_", " ") or url


# ---------- מצב ----------
# busy: מרגע שהתקבלה שאלה ועד שהתשובה מוצגת. כל קלט בזמן הזה נזרק.
# pending: שאלה שכבר נוספה לשיחה ומחכה לתשובה. pending_audio: הקלטה שמחכה לתמלול.
# release: התשובה נשמרה. busy משתחרר רק בתחילת ה-run הבא, אחרי שה-callbacks של אותו run כבר רצו,
# כך שקלט שנשלח בזמן החישוב (Enter כפול, לחיצה על הצעה) מגיע ל-callback כש-busy עדיין True ונזרק.
for _k, _v in {"messages": [], "busy": False, "pending": None, "pending_audio": None,
               "release": False, "audio_n": 0, "last_audio_hash": None}.items():
    st.session_state.setdefault(_k, _v)

VOICE_FAIL_MSG = "לא הצלחתי לשמוע, אפשר לנסות שוב או לכתוב את השאלה."


IL_TZ = ZoneInfo("Asia/Jerusalem")   # השרת בענן רץ ב-UTC


def now_str():
    return datetime.now(IL_TZ).strftime("%H:%M")


def ask(q):
    """נקודת הכניסה היחידה לשאלה (הקלדה, תמלול, הצעה, נסו שוב). בזמן busy או לשאלה ריקה: כלום."""
    q = (q or "").strip()
    if st.session_state.busy or not q:
        return
    st.session_state.messages.append({"role": "user", "content": q, "t": now_str()})
    st.session_state.pending = q
    st.session_state.busy = True


def on_chat_submit():
    ask(st.session_state.get("chat_box"))


def on_audio():
    key = f"audio_{st.session_state.audio_n}"
    audio = st.session_state.get(key)
    if st.session_state.busy or audio is None:
        return
    data = audio.getvalue()
    h = hashlib.md5(data).hexdigest()
    if not data or h == st.session_state.last_audio_hash:   # אותה הקלטה לא מתומללת פעמיים
        return
    st.session_state.last_audio_hash = h
    st.session_state.pending_audio = data
    st.session_state.busy = True
    st.session_state.audio_n += 1   # מפתח חדש: חלון ההקלטה נסגר ומתנקה


def new_chat():
    if st.session_state.busy:
        return
    st.session_state.messages = []


def retry():
    """מוחק את התשובה התקולה ומריץ שוב את השאלה."""
    msgs = st.session_state.messages
    if st.session_state.busy or len(msgs) < 2 or msgs[-1].get("status") != "llm_unavailable":
        return
    q = msgs[-2]["content"]
    del msgs[-2:]
    ask(q)


if st.session_state.release:
    st.session_state.busy = False
    st.session_state.release = False
BUSY = st.session_state.busy


def build_history():
    """היסטוריה לבוט, בלי חילופי דברים שנכשלו בגלל עומס."""
    msgs = st.session_state.messages
    out = []
    for i, m in enumerate(msgs):
        nxt = msgs[i + 1] if i + 1 < len(msgs) else None
        if m["role"] == "assistant" and m.get("status") == "llm_unavailable":
            continue
        if m["role"] == "user" and nxt and nxt.get("status") == "llm_unavailable":
            continue
        out.append({"role": m["role"], "content": m["content"]})
    return out


# ---------- תצוגה ----------
def text_html(text):
    """טקסט רגיל ל-HTML: בטוח, עם שורות חדשות ו-**הדגשה**."""
    s = html.escape(text or "").replace("\n", "<br>")
    s = re.sub(r"(\*\d{3,4}|\b0\d{1,2}-\d{7}\b)", r'<span dir="ltr">\1</span>', s)
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)


def bubble(role, inner, kind="", t=""):
    cls = "user" if role == "user" else "bot"
    tm = f'<div class="time">{t}</div>' if t else ""
    st.markdown(
        f'<div class="row {cls}"><div class="bubble {kind}">{inner}{tm}</div></div>',
        unsafe_allow_html=True,
    )


def sources_html(sources):
    chips = "".join(
        f'<a class="chip" href="{html.escape(u, quote=True)}" target="_blank" rel="noopener">'
        f'<span class="n">{i}</span>{html.escape(source_label(u))}</a>'
        for i, u in enumerate(sources, start=1)
    )
    return f'<div class="sources-title">📎 מקורות באתר הפניקס</div><div class="chips">{chips}</div>'


def render(msg, is_last=False):
    status = msg.get("status")
    t = msg.get("t", "")
    if msg["role"] == "user":
        bubble("user", text_html(msg["content"]), t=t)
        return
    if status == "llm_unavailable":
        bubble("bot", text_html(BUSY_MSG), "warn", t)
        if is_last:
            st.button("🔄 נסו שוב", key="retry_btn", on_click=retry, disabled=BUSY)
        return
    if status == "not_found":
        hint = f'<div class="hint">{text_html(NOT_FOUND_HINT)}</div>'
        bubble("bot", text_html(msg["content"]) + hint, "info", t)
        return
    if status == "personal_only":
        bubble("bot", text_html(msg["content"]), "info", t)
        return
    inner = text_html(msg["content"])
    if msg.get("sources"):
        inner += sources_html(msg["sources"])
    bubble("bot", inner, t=t)


# ---------- כותרת עליונה ----------
logo = logo_data_uri()
logo_html = f'<img src="{logo}" alt="הפניקס"><div class="sep"></div>' if logo else ""
STATUS_ICONS = (
    '<svg width="16" height="11" viewBox="0 0 16 11" fill="#1F2933"><rect x="0" y="7" width="3" height="4" rx="1"/>'
    '<rect x="4.3" y="4.7" width="3" height="6.3" rx="1"/><rect x="8.6" y="2.3" width="3" height="8.7" rx="1"/>'
    '<rect x="12.9" y="0" width="3" height="11" rx="1"/></svg>'
    '<svg width="16" height="12" viewBox="0 0 16 12" fill="none" stroke="#1F2933" stroke-width="1.8" stroke-linecap="round">'
    '<path d="M1.5 4.2a9.5 9.5 0 0 1 13 0"/><path d="M3.8 6.8a6.2 6.2 0 0 1 8.4 0"/><circle cx="8" cy="9.6" r="1" fill="#1F2933"/></svg>'
    '<svg width="25" height="12" viewBox="0 0 25 12" fill="none"><rect x=".7" y=".7" width="21" height="10.6" rx="3" stroke="#1F2933" stroke-width="1.2"/>'
    '<rect x="2.4" y="2.4" width="15" height="7.2" rx="1.6" fill="#1F2933"/><rect x="22.8" y="3.8" width="1.6" height="4.4" rx=".8" fill="#1F2933"/></svg>'
)
st.markdown(
    f'<div class="phonetop"><div class="statusbar"><span>{now_str()}</span><span class="cam"></span>'
    f'<span class="icons">{STATUS_ICONS}</span></div>'
    f'<div class="topbar">{logo_html}<div class="txt"><div class="t1">{APP_TITLE}</div>'
    f'<div class="t2">{APP_SUBTITLE}</div></div></div></div>',
    unsafe_allow_html=True,
)

# ---------- שאלה בקול ----------
# בזמן busy החלון לא מוצג. אחרי כל הקלטה המפתח מתחלף (audio_n), כדי שהחלון ייסגר וההקלטה תתנקה
if not BUSY:
    an = st.session_state.audio_n
    with st.popover(" ", icon=":material/mic:", key=f"voice_pop_{an}"):
        st.caption("לוחצים על המיקרופון, מדברים, ולוחצים שוב לעצירה. השאלה תופיע בצ'אט כפי שהובנה.")
        st.audio_input("הקלטה", key=f"audio_{an}", label_visibility="collapsed", on_change=on_audio)

# ---------- קלט ----------
st.chat_input("יש לך שאלה? אני כאן בשבילך 🙂", key="chat_box", on_submit=on_chat_submit, disabled=BUSY)

msgs = st.session_state.messages

# ---------- מסך פתיחה ----------
if not msgs and not BUSY:
    bubble("bot", text_html(GREETING), t=now_str())
    st.markdown('<div style="height:6px"></div>', unsafe_allow_html=True)
    with st.expander("ℹ️ מה אפשר ומה אי אפשר לשאול"):
        st.markdown(
            "**אפשר:** שאלות על ביטוח, פנסיה, חיסכון והלוואות, איך מגישים תביעה, "
            "אילו מסמכים צריך, ושאלות המשך.\n\n"
            "**אי אפשר:** מידע על הפוליסה או התביעה האישית שלכם, ייעוץ אישי, "
            "או נושאים שאינם באתר."
        )

# ---------- היסטוריה ----------
for i, m in enumerate(msgs):
    render(m, is_last=(i == len(msgs) - 1))

# בועת "הבוט מקליד". החישוב עצמו רץ בסוף הקובץ, אחרי שכל הכפתורים כבר הוצגו נעולים
TYPING_HTML = (
    '<div class="row bot"><div class="bubble"><div class="typing">'
    '<span></span><span></span><span></span></div></div></div>'
)
typing_holder = st.empty()
if BUSY:
    typing_holder.markdown(TYPING_HTML, unsafe_allow_html=True)

# ---------- הצעות לשאלות נוספות ----------
def followup_questions(msgs, n=3):
    asked = {m["content"] for m in msgs if m["role"] == "user"}
    # הצעות שהמודל נתן לתשובה האחרונה (כבר סוננו ב-rag). אם אין, רשימה קבועה לפי נושא
    asked_norm = {rag._norm_q(q) for q in asked}
    own = [q for q in (msgs[-1].get("followups") or []) if rag._norm_q(q) not in asked_norm]
    if own:
        return own[:n]
    last_q = next((m["content"] for m in reversed(msgs) if m["role"] == "user"), "")
    last_a = msgs[-1].get("content", "")
    text = last_q + " " + last_a
    is_car, is_home = bool(CAR_WORDS.search(text)), bool(HOME_WORDS.search(text))
    if is_car and not is_home:
        pools = [FOLLOWUPS_CAR, FOLLOWUPS_OTHER, FOLLOWUPS_APARTMENT]
    elif is_home and not is_car:
        pools = [FOLLOWUPS_APARTMENT, FOLLOWUPS_CAR, FOLLOWUPS_OTHER]
    else:
        pools = [FOLLOWUPS_CAR[:1] + FOLLOWUPS_APARTMENT[:1], FOLLOWUPS_OTHER, FOLLOWUPS_APARTMENT, FOLLOWUPS_CAR]
    out = []
    for pool in pools:
        for q in pool:
            if q not in asked and q not in out:
                out.append(q)
            if len(out) == n:
                return out
    return out


if (
    not BUSY
    and st.session_state.messages
    and st.session_state.messages[-1]["role"] == "assistant"
    and st.session_state.messages[-1].get("status") != "llm_unavailable"
):
    suggestions = followup_questions(st.session_state.messages)
    if suggestions:
        with st.container(key="followups"):
            st.markdown('<div class="followups-title">הצעות לשאלות נוספות:</div>', unsafe_allow_html=True)
            for i, q in enumerate(suggestions):
                st.button(q, key=f"fu_{len(st.session_state.messages)}_{i}", on_click=ask, args=(q,))

# ---------- גלילה אוטומטית לסוף השיחה (בתוך המסגרת) ----------
components.html(
    """<script>
    const box = window.parent.document.querySelector('.block-container');
    if (box) { setTimeout(() => box.scrollTo({top: box.scrollHeight, behavior: 'smooth'}), 150); }
    </script>""",
    height=0,
)

# ---------- תחתית ----------
if st.session_state.messages:
    st.button("✨ שיחה חדשה", on_click=new_chat, key="new_chat", disabled=BUSY)


# ---------- חישוב (רק בזמן busy, אחרי שכל המסך הוצג נעול) ----------
def _bot_msg(res):
    return {
        "role": "assistant",
        "content": res.get("answer", ""),
        "sources": res.get("sources", []),
        "status": res.get("status"),
        "followups": res.get("followups", []),
        "t": now_str(),
    }


if BUSY and st.session_state.pending_audio is not None:
    data, st.session_state.pending_audio = st.session_state.pending_audio, None
    heard = None
    try:
        heard = rag.transcribe(data)
    except Exception as e:
        print("voice error:", e)
    finally:
        heard = (heard or "").strip()
        if heard:   # השאלה נכנסת כהודעת משתמש רגילה, והתשובה תחושב ב-run הבא (busy נשאר)
            st.session_state.messages.append({"role": "user", "content": heard, "t": now_str()})
            st.session_state.pending = heard
        else:
            st.session_state.messages.append(_bot_msg({"answer": VOICE_FAIL_MSG, "status": "meta"}))
            st.session_state.release = True
    st.rerun()

if BUSY and st.session_state.pending:
    question = st.session_state.pending
    history = build_history()[:-1]   # בלי השאלה הנוכחית, שכבר נוספה לשיחה
    res = {"status": "llm_unavailable", "answer": "", "sources": []}
    try:
        res = rag.answer(question, history)   # קריאה אחת בלבד לכל שאלה
    except Exception as e:  # לא מציגים שגיאה טכנית למשתמש
        print("app error:", e)
    finally:   # גם בשגיאה: התשובה נשמרת והנעילה משתחררת
        st.session_state.messages.append(_bot_msg(res))
        st.session_state.pending = None
        st.session_state.release = True
    st.rerun()

if BUSY and not st.session_state.pending and st.session_state.pending_audio is None:
    st.session_state.release = True   # אין מה לחשב (מצב לא צפוי): לא משאירים את הצ'אט נעול
    st.rerun()
st.markdown(f'<div class="disclaimer">{DISCLAIMER}</div>', unsafe_allow_html=True)