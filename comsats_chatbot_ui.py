"""COMSATS University Islamabad — Streamlit chatbot UI.

Retrieval (Chroma + CrossEncoder reranking) loads at startup; the Groq
chat model is created lazily, so the app still opens (with a clear
warning) when GROQ_API_KEY is not set, instead of crashing.
"""

import logging
import os
import re
import time
from pathlib import Path

import dotenv
import streamlit as st
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

from comsats_facts import get_deterministic_facts, get_deterministic_resources

dotenv.load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("comsats_ui")

BASE_DIR = Path(__file__).resolve().parent
CHROMA_DIR = str(BASE_DIR / "chroma_comsats_db")

# Groq retired llama-3.3-70b-versatile for free/developer tiers on
# 2026-08-16 (every call now returns 404 model_not_found). The current
# Groq-recommended replacement is openai/gpt-oss-120b. Override with
# GROQ_MODEL if Groq retires this one too.
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")

# Brand assets (official COMSATS site, verified live Nov 26, 2026).
# The logo is the university seal, upscaled 4x to 748x748 and committed
# to the repo so the app does not depend on comsats.edu.pk being up.
LOGO_URL = "https://raw.githubusercontent.com/abdulsamadxai/comsats-chatbot/main/assets/comsats-logo.png"
CAMPUS_IMG_URL = "https://raw.githubusercontent.com/abdulsamadxai/comsats-chatbot/main/assets/campus-hero.jpg"
ADMISSIONS_URL = "https://admissions.comsats.edu.pk"

st.set_page_config(
    page_title="CUI Assistant | COMSATS University Islamabad",
    page_icon=LOGO_URL,
    layout="centered",
)

# ------------------------------------------------------------------
# Setup (ek hi baar chalta hai, Streamlit cache karta hai)
# ------------------------------------------------------------------


@st.cache_resource
def load_components():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    vector_store = Chroma(
        collection_name="comsats_knowledge",
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return vector_store, reranker


@st.cache_resource
def get_llm():
    """Groq chat model, or None when it cannot be created (no API key,
    or client construction failed)."""
    if not os.environ.get("GROQ_API_KEY"):
        return None
    try:
        return ChatGroq(model=GROQ_MODEL, temperature=0)
    except Exception as e:
        log.warning("Could not create Groq client: %s", e)
        return None


def _collection_count(store) -> int:
    """Number of vectors in the store. Uses Chroma's (private) collection
    handle when available and falls back to the public get() API."""
    try:
        return store._collection.count()
    except Exception:
        return len(store.get()["ids"])


vector_store, reranker = load_components()

if _collection_count(vector_store) == 0:
    st.error(
        "Knowledge base khali hai! Pehle terminal mein 'python comsats_rag.py' "
        "ek dafa chalao taake vector store ban sake, phir is app ko refresh karo."
    )
    st.stop()

if not os.environ.get("GROQ_API_KEY"):
    st.warning(
        "GROQ_API_KEY set nahi hai — AI jawab nahi de sakega. "
        "Apni Groq API key .env file mein ya app ke Secrets mein add karein. "
        "Campus/department/program jaisi fixed maloomat phir bhi mil jayengi."
    )

SYSTEM_PROMPT = (
    "You are CUI Assistant, a helpful chatbot for COMSATS University Islamabad. "
    "Always respond in clear English only, regardless of the language of the question. "
    "You ONLY answer questions related to COMSATS University Islamabad (admissions, "
    "fees, programs, departments, facilities, contacts, scholarships, research, etc). "
    "If the question is NOT about COMSATS University Islamabad, politely say: "
    "'I can only help with questions about COMSATS University Islamabad. "
    "Feel free to ask something else about CUI!' and do not answer it. "
    "For COMSATS-related questions, answer ONLY using the context provided below. "
    "If the answer is not in the context, say 'I couldn't find the exact information, "
    "please confirm from the official COMSATS office or website.' "
    "When discussing admission dates, fee figures, or Fall 2026 admission details, "
    "mention that exact figures may vary by program/batch and recommend checking "
    "the official admission notice for the most current details. "
    "Keep answers short and clear, 2-5 sentences."
)


def retrieve_with_sources(query: str, fetch_k: int = 20, k: int = 8, top_n: int = 6):
    """Top chunks + unke source URLs dono return karta hai. MMR se diverse
    chunks alag-alag pages se aate hain, sirf top-score wale nahi."""
    deterministic = get_deterministic_facts(query)

    results = vector_store.max_marginal_relevance_search(query, k=k, fetch_k=fetch_k)
    top = []
    if results:
        pairs = [[query, doc.page_content] for doc in results]
        scores = reranker.predict(pairs)
        ranked = sorted(zip(results, scores), key=lambda x: x[1], reverse=True)
        top = ranked[:top_n]

    context_parts = []
    if deterministic:
        context_parts.append(deterministic)
    context_parts.extend(doc.page_content for doc, _ in top)
    context = "\n\n---\n\n".join(context_parts)

    # unique source links, best score wale pehle
    seen = set()
    sources = []
    for doc, _ in top:
        url = doc.metadata.get("source", "")
        title = doc.metadata.get("title", url)
        if url and url not in seen:
            seen.add(url)
            sources.append((title, url))

    return context, sources


DOWNLOAD_INTENT_KEYWORDS = [
    "download",
    "prospectus",
    "pamphlet",
    "brochure",
    "pdf",
    "form",
    "link",
    "website",
    "apply online",
    "portal",
]


def wants_source_link(query: str) -> bool:
    q = query.lower()
    # Whole-word match use karo (\b boundaries), warna "form" jaisa chhota
    # keyword "information" ke andar bhi match ho jata hai — galat trigger.
    return any(re.search(rf"\b{re.escape(kw)}\b", q) for kw in DOWNLOAD_INTENT_KEYWORDS)


def ask(query: str):
    query = query.strip()
    if not query:
        return "Please type a question.", []

    # 100% accuracy guarantee: known deterministic facts (campus/department/
    # program lists) LLM se bypass ho ke seedha verbatim return hote hain —
    # koi paraphrase-drift risk nahi.
    deterministic = get_deterministic_facts(query)
    if deterministic:
        return deterministic, []

    try:
        context, sources = retrieve_with_sources(query)
    except Exception as e:
        log.warning("[retrieval error] %s", e)
        return (
            "Sorry, I'm having trouble searching the knowledge base right now. Please try again.",
            [],
        )

    model = get_llm()
    if model is None:
        if not os.environ.get("GROQ_API_KEY"):
            return (
                (
                    "The AI service is not configured yet (GROQ_API_KEY is missing). "
                    "Please add your Groq API key to the app's secrets and try again."
                ),
                [],
            )
        return (
            (
                "Sorry, I couldn't connect to the AI service right now "
                "(could be a rate limit or network issue). Please try again shortly."
            ),
            [],
        )

    prompt = f"""{SYSTEM_PROMPT}

Context:
{context}

Question: {query}
Answer:"""
    try:
        response = None
        last_error = None
        for attempt in range(1, 3):  # 1 try + 1 retry
            try:
                response = model.invoke(prompt)
                break
            except Exception as e:
                last_error = e
                err_str = str(e).lower()
                if attempt == 1 and (
                    "rate" in err_str or "429" in err_str or "timeout" in err_str
                ):
                    time.sleep(3)
                    continue
                raise
        if response is None:
            raise last_error
        answer = response.content
    except Exception as e:
        log.warning("[Groq API error] %s", e)
        answer = (
            "Sorry, I couldn't connect to the AI service right now "
            "(could be a rate limit or network issue). Please try again shortly."
        )
        return answer, []

    # Sources sirf tab return karo jab user ne actually document/link/download manga ho
    relevant_sources = sources if wants_source_link(query) else []

    # Deterministic official resources (jaise Fall 2026 admission notice) hamesha
    # add karo agar topic match kare — chahe download-intent keyword na ho
    for title, url in get_deterministic_resources(query):
        if (title, url) not in relevant_sources:
            relevant_sources.append((title, url))

    return answer, relevant_sources


# ------------------------------------------------------------------
# Streamlit UI
# ------------------------------------------------------------------

# --- COMSATS design (from comsats.edu.pk, checked Nov 26, 2026) ---
# Real brand assets: official logo, Islamabad campus photo, Spring 2027
# admissions. No emojis anywhere — icons are inline SVG.
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@600;700;800&display=swap');
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header[data-testid="stHeader"] {background: transparent;}
    .stApp {background: #f4f7fe; font-family: 'Inter', sans-serif;}
    .block-container {padding-top: 1rem; max-width: 820px;}
    h1, h2, h3, .cui-title, .cui-hero h1, .cui-card b, .cui-stat b {font-family: 'Plus Jakarta Sans', sans-serif;}
    .cui-header {display: flex; align-items: center; gap: 14px; background: rgba(255,255,255,0.92); border: 1px solid #e3e9f7; border-radius: 18px; padding: 12px 18px; box-shadow: 0 2px 10px rgba(15,37,87,0.07); backdrop-filter: blur(8px);}
    .cui-header img {height: 54px; width: auto;}
    .cui-title {font-size: 21px; font-weight: 800; color: #0f2557; line-height: 1.15;}
    .cui-subtitle {font-size: 11px; letter-spacing: 2.5px; color: #5b6b8c; font-weight: 700; margin-top: 2px;}
    .cui-apply {margin-left: auto; background: linear-gradient(135deg, #1d4ed8, #2563eb); color: #ffffff !important; padding: 11px 22px; border-radius: 12px; font-weight: 700; font-size: 14px; text-decoration: none; box-shadow: 0 6px 16px rgba(29,78,216,0.35); white-space: nowrap;}
    .cui-apply:hover {box-shadow: 0 8px 22px rgba(29,78,216,0.45); color: #ffffff !important;}
    .cui-hero {position: relative; border-radius: 24px; overflow: hidden; margin-top: 14px; min-height: 320px; display: flex; align-items: flex-end; box-shadow: 0 12px 32px rgba(10,36,99,0.25);}
    .cui-hero-bg {position: absolute; inset: 0; background: url('https://raw.githubusercontent.com/abdulsamadxai/comsats-chatbot/main/assets/campus-hero.jpg') center bottom/cover no-repeat;}
    .cui-hero-shade {position: absolute; inset: 0; background: linear-gradient(180deg, rgba(7,22,64,0.05) 0%, rgba(7,22,64,0.20) 55%, rgba(7,22,64,0.62) 100%);}
    .cui-hero-inner {position: relative; padding: 30px 28px 26px;}
    .cui-badge {display: inline-flex; align-items: center; background: rgba(255,255,255,0.14); border: 1px solid rgba(255,255,255,0.40); border-radius: 999px; padding: 7px 15px; font-size: 11px; letter-spacing: 2px; font-weight: 700; color: #ffffff; backdrop-filter: blur(4px);}
    .cui-badge-dot {width: 8px; height: 8px; border-radius: 50%; background: #e9a319; display: inline-block; margin-right: 8px; box-shadow: 0 0 0 3px rgba(233,163,25,0.30);}
    .cui-hero h1 {font-size: 42px; line-height: 1.08; font-weight: 800; margin: 16px 0 10px; color: #ffffff;}
    .cui-hero p {color: #dbe6ff; font-size: 15px; line-height: 1.6; margin: 0; max-width: 580px;}
    .cui-stats {display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-top: 12px;}
    .cui-stat {background: #ffffff; border: 1px solid #e3e9f7; border-radius: 16px; padding: 14px 8px; text-align: center; box-shadow: 0 1px 4px rgba(15,37,87,0.05);}
    .cui-stat b {display: block; font-size: 22px; color: #0f2557;}
    .cui-stat span {font-size: 10.5px; letter-spacing: 1.5px; color: #5b6b8c; font-weight: 700;}
    .cui-section {font-size: 13px; letter-spacing: 2px; font-weight: 700; color: #5b6b8c; margin: 22px 0 0;}
    .cui-cards {display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-top: 10px;}
    .cui-card {display: block; background: #ffffff; border: 1px solid #e3e9f7; border-radius: 18px; padding: 18px 14px 16px; text-decoration: none !important; box-shadow: 0 1px 4px rgba(15,37,87,0.05); transition: transform .15s ease, box-shadow .15s ease, border-color .15s ease;}
    .cui-card:hover {transform: translateY(-3px); border-color: #1d4ed8; box-shadow: 0 12px 26px rgba(29,78,216,0.18);}
    .cui-ic {width: 46px; height: 46px; border-radius: 14px; display: flex; align-items: center; justify-content: center; margin-bottom: 11px;}
    .cui-ic svg {width: 24px; height: 24px; stroke: #ffffff; fill: none; stroke-width: 2; stroke-linecap: round; stroke-linejoin: round;}
    .cui-ic-blue {background: linear-gradient(135deg, #1d4ed8, #3b82f6);}
    .cui-ic-violet {background: linear-gradient(135deg, #7c3aed, #a855f7);}
    .cui-ic-teal {background: linear-gradient(135deg, #0d9488, #14b8a6);}
    .cui-ic-amber {background: linear-gradient(135deg, #d97706, #f59e0b);}
    .cui-card b {display: block; font-size: 14px; color: #0f2557;}
    .cui-card span {display: block; font-size: 12px; color: #5b6b8c; margin-top: 3px; line-height: 1.5;}
    [data-testid="stChatMessage"] {background: #ffffff; border: 1px solid #e3e9f7; border-radius: 18px; box-shadow: 0 1px 3px rgba(15,37,87,0.05); padding: 14px 16px;}
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {background: linear-gradient(135deg, #1d4ed8, #2563eb); border-color: #1d4ed8; color: #ffffff;}
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) p {color: #ffffff;}
    [data-testid="stChatInput"] > div {background: #ffffff; border: 1.5px solid #c7d6ff; border-radius: 16px; box-shadow: 0 8px 24px rgba(15,37,87,0.10);}
    .cui-footer {text-align: center; color: #5b6b8c; font-size: 12px; line-height: 1.7; margin-top: 28px; padding-bottom: 12px;}
    .cui-footer em {color: #0f2557;}
    .cui-footer a {color: #1d4ed8; text-decoration: none; font-weight: 600;}
    @media (max-width: 640px) {
      .block-container {padding-left: 12px; padding-right: 12px; padding-top: 0.75rem;}
      .cui-header {gap: 10px; padding: 10px 12px; border-radius: 14px;}
      .cui-header img {height: 40px;}
      .cui-title {font-size: 16px;}
      .cui-subtitle {display: none;}
      .cui-apply {padding: 9px 14px; font-size: 13px; border-radius: 10px;}
      .cui-hero {min-height: 240px; border-radius: 18px; margin-top: 10px;}
      .cui-hero-inner {padding: 20px 18px 18px;}
      .cui-badge {font-size: 10px; letter-spacing: 1.5px; padding: 6px 12px;}
      .cui-hero h1 {font-size: 28px; margin: 12px 0 8px;}
      .cui-hero p {font-size: 13.5px;}
      .cui-stats {grid-template-columns: repeat(2, 1fr); gap: 10px;}
      .cui-stat {padding: 12px 6px;}
      .cui-stat b {font-size: 19px;}
      .cui-cards {grid-template-columns: repeat(2, 1fr); gap: 10px;}
      .cui-card {padding: 14px 12px; border-radius: 16px;}
      .cui-ic {width: 40px; height: 40px; border-radius: 12px; margin-bottom: 9px;}
      .cui-card b {font-size: 13px;}
      .cui-card span {font-size: 11.5px;}
    }
    @media (max-width: 380px) {
      .cui-hero h1 {font-size: 25px;}
      .cui-title {font-size: 15px;}
      .cui-header img {height: 36px;}
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="cui-header">
      <img src="{LOGO_URL}" alt="COMSATS University Islamabad logo" onerror="this.style.display='none'">
      <div>
        <div class="cui-title">COMSATS University Islamabad</div>
        <div class="cui-subtitle">CUI ASSISTANT</div>
      </div>
      <a class="cui-apply" href="{ADMISSIONS_URL}" target="_blank">Apply now</a>
    </div>
    <div class="cui-hero">
      <div class="cui-hero-bg"></div>
      <div class="cui-hero-shade"></div>
      <div class="cui-hero-inner">
        <span class="cui-badge"><span class="cui-badge-dot"></span>ADMISSIONS OPEN &middot; SPRING 2027</span>
        <h1>Ask anything about<br>COMSATS University</h1>
        <p>Admissions, programs, fees, scholarships and campuses &mdash; answered from official COMSATS pages, in seconds.</p>
      </div>
    </div>
    <div class="cui-stats">
      <div class="cui-stat"><b>39,100+</b><span>STUDENTS</span></div>
      <div class="cui-stat"><b>7</b><span>CAMPUSES</span></div>
      <div class="cui-stat"><b>116+</b><span>PROGRAMS</span></div>
      <div class="cui-stat"><b>1998</b><span>FOUNDED</span></div>
    </div>
    """,
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": (
                "Greetings! I am here to provide information about COMSATS University "
                "Islamabad (CUI). Whether you are seeking guidance on admissions, "
                "academic programs, or any other aspect of CUI, I am here to help.\n\n"
                "How may I assist you today?"
            ),
            "sources": [],
        }
    )

# ---- Suggestion cards (plain HTML links with SVG icons) ----
# A card click reloads the page with ?q=<question>; the value is picked
# up here, stored in session state, and the param is cleared so a
# refresh never re-sends the question.
if "pending_suggestion" not in st.session_state:
    st.session_state.pending_suggestion = None
_params = st.query_params
if "q" in _params:
    st.session_state.pending_suggestion = _params["q"]
    st.query_params.clear()
clicked_suggestion = st.session_state.pending_suggestion
st.session_state.pending_suggestion = None

st.markdown(
    """
    <div class="cui-section">START WITH A POPULAR QUESTION</div>
    <div class="cui-cards">
      <a class="cui-card" href="?q=Admissions%20Spring%202027">
        <div class="cui-ic cui-ic-blue"><svg viewBox="0 0 24 24"><path d="M22 10L12 5 2 10l10 5 10-5z"/><path d="M6 12.5V17c0 1.66 2.69 3 6 3s6-1.34 6-3v-4.5"/><path d="M22 10v6"/></svg></div>
        <b>Admissions</b><span>Spring 2027 dates, merit and how to apply</span>
      </a>
      <a class="cui-card" href="?q=Campuses%3F">
        <div class="cui-ic cui-ic-violet"><svg viewBox="0 0 24 24"><path d="M3 21h18"/><path d="M5 21V8l7-5 7 5v13"/><path d="M9 9h.01M9 12h.01M9 15h.01M15 9h.01M15 12h.01M15 15h.01"/></svg></div>
        <b>Campuses</b><span>Seven campuses across Pakistan</span>
      </a>
      <a class="cui-card" href="?q=Departments%3F">
        <div class="cui-ic cui-ic-teal"><svg viewBox="0 0 24 24"><path d="M2 4h6a4 4 0 0 1 4 4v12a3 3 0 0 0-3-3H2z"/><path d="M22 4h-6a4 4 0 0 0-4 4v12a3 3 0 0 1 3-3h7z"/></svg></div>
        <b>Departments</b><span>Faculties and departments at CUI</span>
      </a>
      <a class="cui-card" href="?q=Programs%3F">
        <div class="cui-ic cui-ic-amber"><svg viewBox="0 0 24 24"><circle cx="12" cy="8" r="6"/><path d="M15.5 13.5L17 22l-5-3-5 3 1.5-8.5"/></svg></div>
        <b>Programs</b><span>BS, MS and PhD programs from A to Z</span>
      </a>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---- Show chat history ----
for msg in st.session_state.messages:
    with st.chat_message(
        msg["role"], avatar=LOGO_URL if msg["role"] == "assistant" else None
    ):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            st.markdown("**📎 Source / Download:**")
            for title, url in msg["sources"]:
                st.markdown(f"- [{title}]({url})")

# ---- User input (typed or from a suggestion button) ----
user_input = st.chat_input("Ask me anything") or clicked_suggestion

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant", avatar=LOGO_URL):
        with st.spinner("Thinking..."):
            answer, sources = ask(user_input)
        st.markdown(answer)
        if sources:
            st.markdown("**📎 Source / Download:**")
            for title, url in sources:
                st.markdown(f"- [{title}]({url})")

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )

st.markdown(
    """
    <div class="cui-footer">
      <em>&ldquo;To become a world-class university through excellence in teaching, research, innovation and public service.&rdquo;</em><br>
      Answers come from official COMSATS pages. Confirm dates and fees on the
      <a href="https://admissions.comsats.edu.pk" target="_blank">official admission notice</a>
      or <a href="https://www.comsats.edu.pk" target="_blank">comsats.edu.pk</a>.
    </div>
    """,
    unsafe_allow_html=True,
)
