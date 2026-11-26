import re
import json
import dotenv
import streamlit as st
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from sentence_transformers import CrossEncoder

dotenv.load_dotenv()

# ------------------------------------------------------------------
# Setup (ek hi baar chalta hai, Streamlit cache karta hai)
# ------------------------------------------------------------------

@st.cache_resource
def load_components():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    vector_store = Chroma(
        collection_name="comsats_knowledge",
        embedding_function=embeddings,
        persist_directory="./chroma_comsats_db",
    )
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    model = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    return vector_store, reranker, model


@st.cache_resource
def load_url_to_content():
    """Force-include mechanism ke liye — poora raw scraped content, chunk nahi."""
    try:
        with open("output/all_documents.json", "r", encoding="utf-8") as f:
            raw_docs = json.load(f)
        return {d["url"]: d["content"] for d in raw_docs if d.get("content")}
    except FileNotFoundError:
        return {}


vector_store, reranker, model = load_components()
url_to_content = load_url_to_content()

if vector_store._collection.count() == 0:
    st.error(
        "Knowledge base khali hai! Pehle terminal mein 'python comsats_rag.py' "
        "ek dafa chalao taake vector store ban sake, phir is app ko refresh karo."
    )
    st.stop()

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


# ------------------------------------------------------------------
# Deterministic facts — RAG (vector similarity) enumerable/list-type facts
# ke liye unreliable hoti hai. Ye chand high-value facts hamesha keyword-match
# se guaranteed inject hote hain, retrieval ke upar depend nahi karte.
# ------------------------------------------------------------------
DETERMINISTIC_FACTS = {
    "campus": (
        "COMSATS University Islamabad (CUI) has 7 physical campuses: "
        "Islamabad (Principal Seat, established 1998), Abbottabad (2001), "
        "Wah (2001), Lahore (2002), Attock (2004), Sahiwal (2006), and "
        "Vehari (2008), plus a Virtual Campus (2012) for online/distance learning."
    ),
    "department": (
        "COMSATS University Islamabad (Islamabad Campus) has 13 departments: "
        "Architecture, Art & Design, Computer Science, Electrical Engineering, "
        "Management Sciences, Mathematics, Physics, Chemistry, Bio Sciences, "
        "Humanities, Meteorology, Economics, and Health Informatics."
    ),
    "program": (
        "COMSATS University Islamabad (Islamabad Campus) offers undergraduate (BS), "
        "graduate (MS), and doctoral (PhD) programs spread across its 13 departments: "
        "Architecture, Art & Design, Computer Science, Electrical Engineering, "
        "Management Sciences, Mathematics, Physics, Chemistry, Bio Sciences, "
        "Humanities, Meteorology, Economics, and Health Informatics. The exact "
        "programs offered vary by department — ask about a specific department "
        "(e.g. 'What programs does Computer Science offer?') for details."
    ),
    "scholarship": (
        "For scholarships, please visit: https://ww2.comsats.edu.pk/scholarships/"
    ),
}


def get_deterministic_facts(query: str) -> str:
    import difflib
    q = query.lower()
    words = re.findall(r"[a-z]+", q)
    matched = []
    for kw, fact in DETERMINISTIC_FACTS.items():
        if re.search(rf"\b{kw}", q):
            matched.append(fact)
            continue
        if any(difflib.SequenceMatcher(None, kw, w).ratio() > 0.75 for w in words):
            matched.append(fact)
    return "\n\n".join(matched)


# ------------------------------------------------------------------
# Force-include real content — kuch topics (jaise scholarships) mein RAG
# retrieval consistently weak hoti hai (bohat saari pages, embedding kisi
# ek ko highest score de deta hai baaki chhoot jati hain). Isliye poora
# real scraped content force-include karte hain, guessed fact nahi.
# ------------------------------------------------------------------
FORCE_INCLUDE_TOPICS = {}
MAX_FORCE_INCLUDE_CHARS = 1500


def get_force_included_context(query: str) -> str:
    q = query.lower()
    parts = []
    for pattern, url_substrings in FORCE_INCLUDE_TOPICS.items():
        if not re.search(pattern, q):
            continue
        for sub in url_substrings:
            for url, content in url_to_content.items():
                if sub in url:
                    parts.append(content[:MAX_FORCE_INCLUDE_CHARS])
                    break
    return "\n\n---\n\n".join(parts)


def retrieve_with_sources(query: str, fetch_k: int = 20, k: int = 8, top_n: int = 6):
    """Top chunks + unke source URLs dono return karta hai. MMR se diverse
    chunks alag-alag pages se aate hain, sirf top-score wale nahi."""
    deterministic = get_deterministic_facts(query)
    forced = get_force_included_context(query)

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
    if forced:
        context_parts.append(forced)
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
    "download", "prospectus", "pamphlet", "brochure", "pdf", "form",
    "link", "website", "apply online", "portal",
]


def wants_source_link(query: str) -> bool:
    q = query.lower()
    # Whole-word match use karo (\b boundaries), warna "form" jaisa chhota
    # keyword "information" ke andar bhi match ho jata hai — galat trigger.
    return any(re.search(rf"\b{re.escape(kw)}\b", q) for kw in DOWNLOAD_INTENT_KEYWORDS)


# Kuch high-value OFFICIAL resources jo hamesha dikhne chahiye jab related
# topic poocha jaye — chahe user ne "download"/"link" jaisa word na bola ho.
# Ye scraped text se zyada reliable hain kyunke ye current official notice/ad
# hai (fee structure text mein alag batches ka data mix ho sakta hai aur
# confusing/outdated ho sakta hai).
DETERMINISTIC_RESOURCES = {
    r"\badmission\b.*\bfall\b|\bfall\b.*\badmission\b|\badmissions?\b": [
        ("Fall 2026 Admission Notice (Official)",
         "https://islamabad.comsats.edu.pk/NEFiles/Admission-fall-2026-isb.jpg"),
    ],
}


def get_deterministic_resources(query: str):
    q = query.lower()
    matched = []
    for pattern, resources in DETERMINISTIC_RESOURCES.items():
        if re.search(pattern, q):
            matched.extend(resources)
    return matched


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
        print(f"[retrieval error] {e}")
        return "Sorry, I'm having trouble searching the knowledge base right now. Please try again.", []

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
                if attempt == 1 and ("rate" in err_str or "429" in err_str or "timeout" in err_str):
                    import time
                    time.sleep(3)
                    continue
                raise
        if response is None:
            raise last_error
        answer = response.content
    except Exception as e:
        print(f"[Groq API error] {e}")
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

st.set_page_config(page_title="CUI Assistant", page_icon="🎓")
st.title("🎓 COMSATS University Islamabad")

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant",
        "content": (
            "Greetings! I am here to provide information about COMSATS University "
            "Islamabad (CUI) 🎓. Whether you are seeking guidance on admissions, "
            "academic programs, or any other aspect of CUI, I am here to help.\n\n"
            "How may I assist you today? 😊"
        ),
        "sources": [],
    })

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