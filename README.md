# COMSATS University Islamabad (CUI) Assistant

A Retrieval-Augmented Generation (RAG) chatbot for **COMSATS University Islamabad** built with Streamlit, LangChain, ChromaDB, HuggingFace sentence-transformers, and Groq (`llama-3.3-70b-versatile`).

## Project Structure

- `comsats_chatbot_ui.py` — Streamlit web UI for querying the CUI knowledge base
- `comsats_rag.py` — Vector store ingestion and RAG pipeline (`Chroma` + `CrossEncoder` reranking)
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

3. **Configure environment variables**
   Copy `.env.example` to `.env` and add your API keys:
   ```bash
   cp .env.example .env
   ```
   Required keys:
   - `GROQ_API_KEY`
   - `HF_TOKEN` (optional/recommended for HuggingFace model downloads)

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
