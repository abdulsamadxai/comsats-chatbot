import dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
import json
import re

dotenv.load_dotenv()

# ------------------------------------------------------------------
# 1) Load scraped COMSATS data (all_documents.json wali file se)
# ------------------------------------------------------------------
DATA_FILE = "output/all_documents.json"

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

print(f"Total documents loaded: {len(docs)}")

# URL -> full (unchunked) content lookup — force-include mechanism ke liye
# (RAG chunk-retrieval ke bagair, guaranteed real content pull karne ke liye)
url_to_content = {d["url"]: d["content"] for d in raw_docs if d.get("content")}

# ------------------------------------------------------------------
# 2) Chunking (data bohat zyada hai, isliye chunk size chhota rakha)
# ------------------------------------------------------------------
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
chunks = splitter.split_documents(docs)
print(f"Total chunks created: {len(chunks)}")

# ------------------------------------------------------------------
# 3) Embeddings (multilingual model — English + Roman Urdu queries dono
#    ke liye behtar kaam karta hai, aapke education-chatbot project
#    wale jaisa)
# ------------------------------------------------------------------
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# ------------------------------------------------------------------
# 4) Vector store (Chroma, persistent — sirf pehli dafa build karega)
# ------------------------------------------------------------------
vector_store = Chroma(
    collection_name="comsats_knowledge",
    embedding_function=embeddings,
    persist_directory="./chroma_comsats_db",
)

if vector_store._collection.count() == 0:
    print("Vector store khali hai, chunks add kar raha hun...")
    # ChromaDB ek baar mein max ~5461 documents accept karta hai,
    # isliye batches mein add kar rahe hain (500 per batch)
    batch_size = 500
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        vector_store.add_documents(batch)
        print(f"Added batch {i // batch_size + 1} ({len(batch)} chunks)")
else:
    print("Vector store pehle se mojood hai, dobara build nahi kar raha.")

# ------------------------------------------------------------------
# 4.5) Deterministic facts — kuch cheezein (poori list/count) RAG se
#      reliably nahi milti kyunke vector similarity 'fixed enumerable
#      facts' ke liye consistent nahi hoti (chhota fact document bade
#      documents ke against compete karke haar sakta hai). Isliye in
#      chand high-value facts ko keyword-match se hamesha guaranteed
#      inject karte hain, RAG ke upar depend nahi karte.
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
        # Pehle exact/prefix match try karo (fast)
        if re.search(rf"\b{kw}", q):
            matched.append(fact)
            continue
        # Fallback: typo-tolerant fuzzy match (jaise "shcolorships",
        # "scholorships" — "scholarship" spelling exact na ho tab bhi match ho)
        if any(difflib.SequenceMatcher(None, kw, w).ratio() > 0.75 for w in words):
            matched.append(fact)
    return "\n\n".join(matched)


# ------------------------------------------------------------------
# 4.6) Force-include real content — kuch topics (jaise scholarships) mein
#      RAG retrieval consistently weak/unreliable hai (bohat saari alag
#      pages hain, embedding kisi ek ko highest score de deta hai baaki
#      chhoot jati hain). Isliye in topics ke liye hum poora real scraped
#      content force-include karte hain (guessed/hardcoded fact nahi,
#      asli scraped page ka text), taake accuracy guaranteed rahe.
# ------------------------------------------------------------------
FORCE_INCLUDE_TOPICS = {}
MAX_FORCE_INCLUDE_CHARS = 1500  # per document, taake prompt bohat bara na ho jaye


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


# Kuch high-value OFFICIAL resources jo hamesha jawab ke sath dikhne
# chahiye jab related topic poocha jaye — chahe download-intent word na ho.
DETERMINISTIC_RESOURCES = {
    r"\badmission\b.*\bfall\b|\bfall\b.*\badmission\b|\badmissions?\b": [
        "https://islamabad.comsats.edu.pk/NEFiles/Admission-fall-2026-isb.jpg",
    ],
}


def get_deterministic_resources(query: str) -> list:
    q = query.lower()
    matched = []
    for pattern, urls in DETERMINISTIC_RESOURCES.items():
        if re.search(pattern, q):
            matched.extend(urls)
    return matched


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
    correct rahenge. Force-included real content (scholarships jaisi
    weak-retrieval topics ke liye) bhi isi tarah guaranteed include hota hai.
    """
    deterministic = get_deterministic_facts(query)
    forced = get_force_included_context(query)

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
    if forced:
        parts.append(forced)
    parts.extend(top_docs)

    return "\n\n---\n\n".join(parts)


# ------------------------------------------------------------------
# 6) Groq model + simple prompt
# ------------------------------------------------------------------
model = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

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
        print(f"[retrieval error] {e}")
        return "Sorry, mujhe knowledge base search karne mein masla aa raha hai. Thodi dair baad try karein."

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
                if attempt == 1 and ("rate" in err_str or "429" in err_str or "timeout" in err_str):
                    import time
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
            links_text = "\n".join(f"- {url}" for url in resource_links)
            answer = f"{answer}\n\nFor more details:\n{links_text}"

        return answer
    except Exception as e:
        # Groq rate limit, network issue, ya koi bhi API error yahan catch hoga —
        # taake pura app crash na ho, user ko friendly message mile
        print(f"[Groq API error] {e}")
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
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("CUI Assistant: Thanks for using CUI Assistant! Best of luck with COMSATS. 🎓")
            break

        answer = ask(user_input)
        print("CUI Assistant:", answer)