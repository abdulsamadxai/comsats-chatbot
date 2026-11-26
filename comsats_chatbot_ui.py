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

st.set_page_config(
    page_title="CUI Assistant | COMSATS University Islamabad",
    page_icon="🎓",
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

# --- COMSATS portal styling (institutional blue, like admissions.comsats.edu.pk) ---
LOGO_URL = "https://www.comsats.edu.pk/images/logo.png"
ADMISSIONS_URL = "https://admissions.comsats.edu.pk"

st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header[data-testid="stHeader"] {background: transparent;}
    .stApp {background: #f5f7fd;}
    .block-container {padding-top: 1.2rem; max-width: 780px;}
    .cui-header {display: flex; align-items: center; gap: 14px; background: #ffffff; border: 1px solid #e3e9f7; border-radius: 16px; padding: 12px 18px; box-shadow: 0 1px 4px rgba(15, 37, 87, 0.06);}
    .cui-header img {height: 54px; width: auto;}
    .cui-title {font-size: 21px; font-weight: 800; color: #0f2557; line-height: 1.15;}
    .cui-subtitle {font-size: 11px; letter-spacing: 2.5px; color: #5b6b8c; font-weight: 700; margin-top: 2px;}
    .cui-apply {margin-left: auto; background: #1d4ed8; color: #ffffff !important; padding: 10px 20px; border-radius: 10px; font-weight: 700; font-size: 14px; text-decoration: none; box-shadow: 0 4px 14px rgba(29, 78, 216, 0.35); white-space: nowrap;}
    .cui-apply:hover {background: #1e40af; color: #ffffff !important;}
    .cui-hero {background: linear-gradient(115deg, #0a2a6b 0%, #123a8f 55%, #1d4ed8 100%); border-radius: 20px; padding: 30px 28px 26px; color: #ffffff; margin-top: 14px; position: relative; overflow: hidden;}
    .cui-hero-dots {position: absolute; inset: 0; background-image: radial-gradient(rgba(255,255,255,0.28) 1.2px, transparent 1.2px); background-size: 22px 22px; opacity: 0.35;}
    .cui-hero-inner {position: relative;}
    .cui-badge {display: inline-flex; align-items: center; background: rgba(255,255,255,0.12); border: 1px solid rgba(255,255,255,0.35); border-radius: 999px; padding: 6px 14px; font-size: 11px; letter-spacing: 2px; font-weight: 700;}
    .cui-badge-dot {width: 8px; height: 8px; border-radius: 50%; background: #22c55e; display: inline-block; margin-right: 8px;}
    .cui-hero h1 {font-size: 40px; line-height: 1.08; font-weight: 800; margin: 16px 0 10px; color: #ffffff;}
    .cui-hero p {color: #c7d6ff; font-size: 15px; line-height: 1.55; margin: 0; max-width: 560px;}
    [data-testid="stChatMessage"] {background: #ffffff; border: 1px solid #e3e9f7; border-radius: 16px; box-shadow: 0 1px 3px rgba(15, 37, 87, 0.05);}
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {background: #1d4ed8; border-color: #1d4ed8; color: #ffffff;}
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) p {color: #ffffff;}
    [data-testid="stButton"] > button {background: #ffffff; color: #1d4ed8; border: 1.5px solid #c7d6ff; border-radius: 999px; font-weight: 600; box-shadow: 0 1px 3px rgba(15, 37, 87, 0.06);}
    [data-testid="stButton"] > button:hover {background: #1d4ed8; color: #ffffff; border-color: #1d4ed8;}
    [data-testid="stButton"] > button:focus {box-shadow: 0 0 0 3px rgba(29, 78, 216, 0.25); outline: none;}
    .cui-footer {text-align: center; color: #5b6b8c; font-size: 12px; margin-top: 26px; padding-bottom: 10px;}
    .cui-footer a {color: #1d4ed8; text-decoration: none; font-weight: 600;}
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
      <div class="cui-hero-dots"></div>
      <div class="cui-hero-inner">
        <span class="cui-badge"><span class="cui-badge-dot"></span>ADMISSIONS OPEN AT 7 OF 7 CAMPUSES</span>
        <h1>Become Pakistan's<br>next great COMSIAN</h1>
        <p>Ask about admissions, programs, fees, scholarships and campuses at seven campuses of a top-ranked research university. One question, one clear answer.</p>
      </div>
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
                "Islamabad (CUI) 🎓. Whether you are seeking guidance on admissions, "
                "academic programs, or any other aspect of CUI, I am here to help.\n\n"
                "How may I assist you today? 😊"
            ),
            "sources": [],
        }
    )

# ---- Quick suggestion buttons ----
st.write("**Quick suggestions:**")
cols = st.columns(4)
suggestions = ["Admissions FALL 2026", "Campuses?", "Departments?", "Programs?"]
clicked_suggestion = None
for col, s in zip(cols, suggestions):
    if col.button(s, use_container_width=True):
        clicked_suggestion = s

# ---- Show chat history ----
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
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

    with st.chat_message("assistant"):
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
      Answers come from official COMSATS pages. Confirm dates and fees on the
      <a href="https://admissions.comsats.edu.pk" target="_blank">official admission notice</a>
      or <a href="https://www.comsats.edu.pk" target="_blank">comsats.edu.pk</a>.
    </div>
    """,
    unsafe_allow_html=True,
)
