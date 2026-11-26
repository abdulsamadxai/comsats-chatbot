# COMSATS University Islamabad (CUI) Assistant

A Retrieval-Augmented Generation (RAG) chatbot for **COMSATS University Islamabad** built with Streamlit, LangChain, ChromaDB, HuggingFace sentence-transformers, and Groq (`openai/gpt-oss-120b`).

## Project Structure

- `comsats_chatbot_ui.py` — Streamlit web UI for querying the CUI knowledge base
- `comsats_rag.py` — Vector store ingestion and RAG pipeline (`Chroma` + `CrossEncoder` reranking), plus a CLI chat loop
- `comsats_facts.py` — Shared deterministic facts and official resource links (used by both entry points)
- `curated_scrape.py` — Curated scraper for official COMSATS pages into `output/all_documents.json`
- `output/all_documents.json` — Pre-scraped curated COMSATS knowledge documents
- `chroma_comsats_db/` — Pre-built ChromaDB vector store (ready to run without re-scraping)
- `.env.example` — Template for required environment variables

## Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/abdulsamadxai/comsats-chatbot.git
   cd comsats-chatbot
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   On first run the app downloads two small HuggingFace models
   (embedding + reranker, ~200 MB total) — this needs internet access
   and can take a few minutes.

3. **Configure environment variables**
   Copy `.env.example` to `.env` and add your API keys:
   ```bash
   cp .env.example .env
   ```
   Required keys:
   - `GROQ_API_KEY` — Groq API key (free at https://console.groq.com/keys).
     Without it the app still opens, but AI answers are disabled and a
     clear warning is shown; fixed facts (campuses, departments, programs)
     still work.
   - `HF_TOKEN` (optional/recommended for HuggingFace model downloads)
   - `GROQ_MODEL` (optional) — override the Groq chat model if Groq
     retires the default. Default: `openai/gpt-oss-120b`.

4. **Run the chatbot UI**
   ```bash
   streamlit run comsats_chatbot_ui.py
   ```

## Rebuilding the Knowledge Base (Optional)

If you want to refresh the scraped data and vector store:
```bash
python curated_scrape.py
python comsats_rag.py
```
Rebuilding the vector store does **not** need a `GROQ_API_KEY` — the
chat model is only created when a question is actually answered.
To rebuild the store from scratch, delete `chroma_comsats_db/` first
and then run `python comsats_rag.py`.

## Deploying

Streamlit Community Cloud (free) is the simplest host: connect the repo,
set the app file to `comsats_chatbot_ui.py`, and add `GROQ_API_KEY` in
the app's Secrets. No other secrets are required.
