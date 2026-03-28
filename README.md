# RAG Application (Literature Assistant)

A Retrieval-Augmented Generation (RAG) app for PDF literature analysis with:
- **Q&A with citations**
- **Fact-checking against retrieved evidence**
- **Literature review drafting**
- **Evaluation pipeline for retrieval + generation quality**

Built with **Streamlit**, **LangChain**, **ChromaDB**, and **Gemini models**.

---

## Features

- Ingests PDFs from `Data/`
- Builds and queries a persisted Chroma vector store in `vectorstore/chroma`
- Adds source-aware citations to model outputs
- Supports both:
  - **Web app** (`streamlit_app.py`)
  - **CLI workflow** (`step1` → `step5` scripts)

---

## Project Structure

```text
.
├── Data/                          # Input PDFs
├── vectorstore/chroma/            # Persisted Chroma index
├── streamlit_app.py               # Streamlit UI
├── rag_cli.py                     # Shared RAG logic + routing
├── step1_generate_embeddings.py   # Build vectorstore from PDFs
├── step2_qa_with_citations.py     # Q&A mode
├── step3_fact_check.py            # Fact-check mode
├── step4_literature_review.py     # Review drafting mode
├── step5_evaluate_rag.py          # Evaluation script
├── eval_dataset.jsonl             # Evaluation dataset (JSONL)
├── requirements.txt
└── .env.example
```

---

## Prerequisites

- Python 3.10+
- A valid **Google Gemini API key**

---

## Local Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create `.env` from template and add your API key:

```bash
cp .env.example .env
```

`.env` should contain:

```env
GOOGLE_API_KEY=your_actual_key_here
```

---

## Run the Streamlit App

```bash
streamlit run streamlit_app.py
```

In the sidebar, verify:
- Vector store path: `vectorstore/chroma`
- Embedding model: `models/gemini-embedding-001`

---

## CLI Workflow

### Step 1: Build / Rebuild embeddings

```bash
python step1_generate_embeddings.py --rebuild
```

### Step 2: Ask questions with citations

```bash
python step2_qa_with_citations.py --question "What are the main findings?"
```

### Step 3: Fact-check a claim

```bash
python step3_fact_check.py --text "Your claim text here"
```

### Step 4: Draft a literature review

```bash
python step4_literature_review.py --outline "Theme 1; Theme 2; Theme 3"
```

### Step 5: Evaluate RAG quality

```bash
python step5_evaluate_rag.py --eval-set eval_dataset.jsonl --output rag_eval_results.json
```

Optional tuning (higher precision with stable recall):

```bash
python step5_evaluate_rag.py --top-k 4 --candidate-multiplier 5
```

---

## Deployment (Streamlit Community Cloud)

1. Push this repository to GitHub.
2. In Streamlit Community Cloud, create a new app from this repo.
3. Set:
   - **Main file path**: `streamlit_app.py`
4. Add secret in Streamlit app settings:

```toml
GOOGLE_API_KEY="your_actual_key_here"
```

5. Deploy.

### Notes

- `vectorstore/chroma` is already included in this repo, so the app can start without re-embedding.
- If you change PDFs in `Data/`, regenerate embeddings using Step 1 and push updated `vectorstore/chroma`.

---

## Troubleshooting

- **“GOOGLE_API_KEY is missing”**
  - Ensure `.env` exists locally, or Streamlit secret is set in cloud.

- **“Vector store not found at vectorstore/chroma”**
  - Make sure `vectorstore/chroma` is present in the repo or run Step 1.

- **Model access errors**
  - Verify API key permissions and model availability in your Google AI account.

---

## License

For personal/research use. Add a formal license file if needed.
