"""COMSATS University Islamabad — RAG pipeline + CLI chat.

Vector store ingestion (Chroma + CrossEncoder reranking) and the
question-answering flow. The Groq chat model is created lazily, so
building/refreshing the vector store does NOT need a GROQ_API_KEY —
only answering non-deterministic questions does.

Usage:
    python comsats_rag.py        # build the vector store if empty, then chat
"""

import json
import logging
import os
import time
from pathlib import Path

import dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder

from comsats_facts import get_deterministic_facts, get_deterministic_resources

dotenv.load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("comsats_rag")

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "output" / "all_documents.json"
CHROMA_DIR = str(BASE_DIR / "chroma_comsats_db")

# Groq retired llama-3.3-70b-versatile for free/developer tiers on
# 2026-08-16 (every call now returns 404 model_not_found). The current
# Groq-recommended replacement is openai/gpt-oss-120b. Override with
# GROQ_MODEL if Groq retires this one too.
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")

# ------------------------------------------------------------------
# 1) Load scraped COMSATS data (all_documents.json wali file se)
# ------------------------------------------------------------------
try:
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        raw_docs = json.load(f)
except FileNotFoundError:
    raise SystemExit(
        f"ERROR: '{DATA_FILE}' nahi mili. Pehle 'python curated_scrape.py' "
        f"chalao taake ye file ban sake, phir dobara try karo."
    )

docs = []
for d in raw_docs:
    if d.get("content"):
        docs.append(
            Document(
                page_content=d["content"],
                metadata={
                    "source": d["url"],
                    "title": d.get("title", ""),
                    "category": d.get("category", "general"),
                },
            )
        )

if not docs:
    raise SystemExit(
        f"ERROR: '{DATA_FILE}' khali hai ya koi valid content nahi mila. "
        f"Scraper dobara chalao aur check karo koi errors to nahi aaye."
    )

log.info("Total documents loaded: %d", len(docs))

# ------------------------------------------------------------------
# 2) Chunking (data bohat zyada hai, isliye chunk size chhota rakha)
# ------------------------------------------------------------------
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
chunks = splitter.split_documents(docs)
log.info("Total chunks created: %d", len(chunks))

# ------------------------------------------------------------------
# 3) Embeddings (multilingual model — English + Roman Urdu queries dono
#    ke liye behtar kaam karta hai, aapke education-chatbot project
#    wale jaisa)
# ------------------------------------------------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# ------------------------------------------------------------------
# 4) Vector store (Chroma, persistent — sirf pehli dafa build karega)
# ------------------------------------------------------------------
vector_store = Chroma(
    collection_name="comsats_knowledge",
    embedding_function=embeddings,
    persist_directory=CHROMA_DIR,
)


def _collection_count(store) -> int:
    """Number of vectors in the store. Uses Chroma's (private) collection
    handle when available and falls back to the public get() API."""
    try:
        return store._collection.count()
    except Exception:
        return len(store.get()["ids"])


if _collection_count(vector_store) == 0:
    log.info("Vector store khali hai, chunks add kar raha hun...")
    # ChromaDB ek baar mein max ~5461 documents accept karta hai,
    # isliye batches mein add kar rahe hain (500 per batch)
    batch_size = 500
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        vector_store.add_documents(batch)
        log.info("Added batch %d (%d chunks)", i // batch_size + 1, len(batch))
else:
    log.info("Vector store pehle se mojood hai, dobara build nahi kar raha.")

# ------------------------------------------------------------------
# 5) Reranker (top-k results ko query ke sath best-match order mein lata hai)
# ------------------------------------------------------------------
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def retrieve_context(query: str, fetch_k: int = 20, k: int = 8, top_n: int = 6) -> str:
    """
    MMR (Maximal Marginal Relevance) search use karta hai — sirf 'sabse
    similar' chunks nahi, balke DIVERSE chunks alag-alag source pages se
    lata hai. Ye zaroori hai jab data multiple pages mein spread ho
    (jaise 13 alag department pages) — warna list-type questions
    ('Departments?', 'Scholarships?') mein sirf 1-2 pages ka data aata
    hai, baaki chhoot jata hai.

    Deterministic facts (agar query match kare) hamesha context mein
    upar add hote hain — RAG retrieval fail ho bhi jaye to ye guaranteed
    correct rahenge.
    """
    deterministic = get_deterministic_facts(query)

    results = vector_store.max_marginal_relevance_search(query, k=k, fetch_k=fetch_k)
    top_docs = []
    if results:
        pairs = [[query, doc.page_content] for doc in results]
        scores = reranker.predict(pairs)
        ranked = sorted(zip(results, scores), key=lambda x: x[1], reverse=True)
        top_docs = [doc.page_content for doc, score in ranked[:top_n]]

    parts = []
    if deterministic:
        parts.append(deterministic)
    parts.extend(top_docs)

    return "\n\n---\n\n".join(parts)


# ------------------------------------------------------------------
# 6) Groq model — lazily created (sirf jab pehli dafa zaroorat ho).
#    Isse vector-store build/import bina GROQ_API_KEY ke bhi chalta hai.
# ------------------------------------------------------------------
_model = None


def get_model():
    """Return the Groq chat model, or None when it cannot be created
    (no API key, or client construction failed)."""
    global _model
    if _model is not None:
        return _model
    if not os.environ.get("GROQ_API_KEY"):
        return None
    try:
        _model = ChatGroq(model=GROQ_MODEL, temperature=0)
    except Exception as e:
        log.warning("Could not create Groq client: %s", e)
        return None
    return _model


SYSTEM_PROMPT = (
    "You are CUI Assistant, a helpful chatbot for COMSATS University Islamabad. "
    "You ONLY answer questions related to COMSATS University Islamabad (admissions, "
    "fees, programs, departments, facilities, contacts, scholarships, research, etc). "
    "If the question is NOT about COMSATS University Islamabad (e.g. general knowledge, "
    "other topics), politely say: 'Main sirf COMSATS University Islamabad ke bare mein "
    "madad kar sakta hun. Koi aur COMSATS-related sawal poochein!' and do not answer it. "
    "For COMSATS-related questions, answer ONLY using the context provided below. "
    "If the answer is not in the context, say 'Mujhe exact information nahi mili, "
    "please confirm from the official COMSATS office or website.' "
    "Keep answers short and clear, 2-5 sentences."
)


def ask(query: str) -> str:
    query = query.strip()
    if not query:
        return "Please type a question."

    # 100% accuracy guarantee: agar query specifically ek known deterministic
    # fact (campus/department/program list) se match karti hai, to LLM ko
    # bilkul call nahi karte — seedha verified fact verbatim return karte hain.
    # Isse LLM ke paraphrase karte waqt koi bhi drift/error ka risk khatam
    # ho jata hai in specific facts ke liye.
    deterministic = get_deterministic_facts(query)
    if deterministic:
        return deterministic

    try:
        context = retrieve_context(query)
    except Exception as e:
        log.warning("[retrieval error] %s", e)
        return "Sorry, mujhe knowledge base search karne mein masla aa raha hai. Thodi dair baad try karein."

    model = get_model()
    if model is None:
        if not os.environ.get("GROQ_API_KEY"):
            return (
                "AI service configure nahi hai: GROQ_API_KEY set nahi hai. "
                ".env file mein apni Groq API key add karein, phir dobara try karein."
            )
        return (
            "Sorry, is waqt AI service se connect nahi ho pa raha "
            "(rate limit ya network issue ho sakta hai). Thodi dair baad try karein."
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
                # Rate limit / transient errors ke liye chhoti si wait ke baad
                # ek dafa retry karo — permanent errors (auth, bad request)
                # pe retry se koi fayda nahi, seedha fail hone do
                if attempt == 1 and (
                    "rate" in err_str or "429" in err_str or "timeout" in err_str
                ):
                    time.sleep(3)
                    continue
                raise
        if response is None:
            raise last_error
        answer = response.content

        # Deterministic resource links (jaise scholarships/admission page)
        # hamesha answer ke sath append karo agar topic match kare
        resource_links = get_deterministic_resources(query)
        if resource_links:
            links_text = "\n".join(f"- {url}" for _, url in resource_links)
            answer = f"{answer}\n\nFor more details:\n{links_text}"

        return answer
    except Exception as e:
        # Groq rate limit, network issue, ya koi bhi API error yahan catch hoga —
        # taake pura app crash na ho, user ko friendly message mile
        log.warning("[Groq API error] %s", e)
        return (
            "Sorry, is waqt AI service se connect nahi ho pa raha "
            "(rate limit ya network issue ho sakta hai). Thodi dair baad try karein."
        )


# ------------------------------------------------------------------
# 7) Chat loop
# ------------------------------------------------------------------
if __name__ == "__main__":
    print(
        "CUI Assistant: Greetings! I am here to provide information about "
        "COMSATS University Islamabad (CUI) 🎓. Whether you are seeking guidance "
        "on admissions, academic programs, or any other aspect of CUI, I am here "
        "to help.\nHow may I assist you today? 😊"
    )

    while True:
        try:
            user_input = input("You: ")
        except (EOFError, KeyboardInterrupt):
            print(
                "\nCUI Assistant: Thanks for using CUI Assistant! Best of luck with COMSATS. 🎓"
            )
            break
        if user_input.lower() == "exit":
            print(
                "CUI Assistant: Thanks for using CUI Assistant! Best of luck with COMSATS. 🎓"
            )
            break

        answer = ask(user_input)
        print("CUI Assistant:", answer)
