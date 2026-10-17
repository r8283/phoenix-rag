"""
app.py - ממשק הצ'אט (Streamlit).
הרצה:  python -m streamlit run app.py
לוגו (לא חובה): assets/logo.png
"""
from pathlib import Path

import streamlit as st

import rag

LOGO = Path("assets/logo.png")
HAS_LOGO = LOGO.exists()

APP_TITLE = "עוזר דיגיטלי לשאלות על ביטוח"
APP_SUBTITLE = "התשובות מבוססות על תוכן אתר הפניקס בלבד, עם קישור למקור"
DISCLAIMER = (
    "פרויקט האקתון. השירות אינו אתר רשמי של הפניקס, אינו ייעוץ ביטוחי "
    "ואינו מחליף פנייה לשירות הלקוחות."
)
SITE_URL = "https://www.fnx.co.il"

SUGGESTIONS = [
    "מה זה ביטוח תכולה?",
    "איך מגישים תביעה בביטוח רכב?",
    "מה עושים אחרי פריצה לבית?",
    "אילו מסמכים צריך לצרף לתביעה בביטוח דירה?",
]

BUSY_MSG = "השירות עמוס כרגע. אפשר לנסות שוב בעוד רגע."
NOT_FOUND_HINT = (
    "אפשר לנסות לנסח את השאלה אחרת, או לפנות לשירות הלקוחות של הפניקס."
)

st.set_page_config(
    page_title=APP_TITLE,
    page_icon=str(LOGO) if HAS_LOGO else "💬",
    layout="centered",
)

# ---------- עיצוב ----------
# אפשר לשנות את הצבעים כאן
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Heebo:wght@400;500;700&display=swap');
:root { --brand: #E4572E; --brand-soft: #FDEEE8; --ink: #1F2933; --muted: #6B7280; }
html, body, .stApp, [data-testid="stSidebar"] { direction: rtl; font-family: 'Heebo', sans-serif; }
.stApp { background: #FAFAF8; }
[data-testid="stSidebar"] { background: #FFFFFF; border-left: 1px solid #ECECEC; }
.block-container { padding-top: 2rem; max-width: 760px; }
.hero { display:flex; align-items:center; gap:14px; margin-bottom: .4rem; }
.hero h1 { font-size: 1.6rem; margin: 0; color: var(--ink); font-weight: 700; }
.hero p { margin: 2px 0 0 0; color: var(--muted); font-size: .95rem; }
[data-testid="stChatMessage"] { direction: rtl; border-radius: 14px; padding: .9rem 1rem; }
.stButton > button { border-radius: 12px; border: 1px solid #E5E7EB; text-align: right; }
.stButton > button:hover { border-color: var(--brand); color: var(--brand); }
.disclaimer { color: var(--muted); font-size: .8rem; text-align: center; margin-top: 1.2rem; }
.sources-title { color: var(--muted); font-size: .85rem; margin-top: .6rem; }
a { color: var(--brand) !important; }
</style>
""",
    unsafe_allow_html=True,
)

# ---------- מצב ----------
if "messages" not in st.session_state:
    st.session_state.messages = []  # {"role","content","sources","status"}


def ask(q):
    st.session_state.pending = q


def new_chat():
    st.session_state.messages = []


def retry():
    """מוחק את התשובה התקולה ומריץ שוב את השאלה."""
    msgs = st.session_state.messages
    if len(msgs) >= 2 and msgs[-1].get("status") == "llm_unavailable":
        q = msgs[-2]["content"]
        del msgs[-2:]
        st.session_state.pending = q


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


def source_label(url):
    slug = url.rstrip("/").split("/")[-1]
    return slug.replace("-", " ").replace("_", " ") or url


# ---------- תצוגה ----------
def render_body(msg, is_last=False):
    status = msg.get("status")
    if msg["role"] == "user":
        st.markdown(msg["content"])
        return
    if status == "llm_unavailable":
        st.warning(BUSY_MSG)
        if is_last:
            st.button("נסו שוב", key="retry_btn", on_click=retry)
        return
    if status in ("not_found", "personal_only"):
        st.info(msg["content"] + ("\n\n" + NOT_FOUND_HINT if status == "not_found" else ""))
        return
    st.markdown(msg["content"])
    if msg.get("sources"):
        st.markdown('<div class="sources-title">📎 מקורות באתר</div>', unsafe_allow_html=True)
        for i, url in enumerate(msg["sources"], start=1):
            st.markdown(f"{i}. [{source_label(url)}]({url})")


def avatar_for(role):
    if role == "assistant":
        return str(LOGO) if HAS_LOGO else "💬"
    return "🙂"


def render(msg, is_last=False):
    with st.chat_message(msg["role"], avatar=avatar_for(msg["role"])):
        render_body(msg, is_last)


# ---------- סרגל צד ----------
with st.sidebar:
    if HAS_LOGO:
        st.image(str(LOGO), width=140)
    st.markdown("### מה אפשר לשאול")
    st.markdown(
        "- מה כוללת ומה לא כוללת פוליסה\n"
        "- איך מגישים תביעה ואילו מסמכים צריך\n"
        "- מה עושים אחרי אירוע (פריצה, הצפה ועוד)\n"
        "- שאלות המשך על מה שנאמר"
    )
    st.markdown("### מה לא אפשר")
    st.markdown(
        "- מידע על הפוליסה או התביעה האישית שלכם\n"
        "- ייעוץ אישי או המלצה מה לבחור\n"
        "- נושאים שאינם מופיעים באתר"
    )
    st.button("שיחה חדשה", on_click=new_chat, use_container_width=True)
    st.caption(f"האתר המלא: [{SITE_URL.replace('https://', '')}]({SITE_URL})")

# ---------- כותרת ----------
st.markdown(
    f'<div class="hero"><div><h1>{APP_TITLE}</h1><p>{APP_SUBTITLE}</p></div></div>',
    unsafe_allow_html=True,
)

# ---------- קלט ----------
typed = st.chat_input("כתבו כאן את השאלה...")
question = typed or st.session_state.pop("pending", None)

# ---------- מסך פתיחה ----------
if not st.session_state.messages and not question:
    st.markdown("#### שלום, במה אפשר לעזור?")
    st.write("אפשר לשאול על ביטוחים והגשת תביעות. כל תשובה מגיעה עם קישור לעמוד באתר.")
    cols = st.columns(2)
    for i, s in enumerate(SUGGESTIONS):
        cols[i % 2].button(s, key=f"sugg_{i}", on_click=ask, args=(s,), use_container_width=True)

# ---------- היסטוריה ----------
msgs = st.session_state.messages
for i, m in enumerate(msgs):
    render(m, is_last=(i == len(msgs) - 1) and not question)

# ---------- שאלה חדשה ----------
if question:
    history = build_history()
    user_msg = {"role": "user", "content": question}
    st.session_state.messages.append(user_msg)
    render(user_msg)

    with st.chat_message("assistant", avatar=avatar_for("assistant")):
        with st.spinner("מחפש בתוכן האתר..."):
            try:
                res = rag.answer(question, history)
            except Exception as e:  # לא מציגים שגיאה טכנית למשתמש
                print("app error:", e)
                res = {"status": "llm_unavailable", "answer": "", "sources": []}
        bot_msg = {
            "role": "assistant",
            "content": res.get("answer", ""),
            "sources": res.get("sources", []),
            "status": res.get("status"),
        }
        st.session_state.messages.append(bot_msg)
        render_body(bot_msg, is_last=True)

st.markdown(f'<div class="disclaimer">{DISCLAIMER}</div>', unsafe_allow_html=True)