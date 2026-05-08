# RAG Pipeline – Medical Document Q&A

A Retrieval-Augmented Generation (RAG) system that ingests medical PDF documents, stores embeddings in a ChromaDB vector database, and exposes a FastAPI `/ask` endpoint for question answering with source attribution.

---

## Project Structure

```
├── deploy.py                  # FastAPI server with /ask endpoint
├── retrieval_pipeline.py      # Query logic: embedding, retrieval, LLM generation
├── ingestion_pipeline.ipynb   # Document ingestion & chunking notebook
├── requirements.txt           # Python dependencies
├── .env                       # Environment variables (API keys, K)
├── docs/                      # Source PDF documents
│   └── diabetes_*.pdf
└── db/
    └── chroma_db/             # Persisted ChromaDB vector store
```

---

## 1. Dependency Installation

### Prerequisites

- Python 3.10+
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) installed and on `PATH` (required by `unstructured` for PDF parsing)

### Install Python dependencies

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate

pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in the project root:

```env
HUGGINGFACEHUB_API_TOKEN=<your-huggingface-api-token>
K=5
```

| Variable                   | Description                                                   |
| -------------------------- | ------------------------------------------------------------- |
| `HUGGINGFACEHUB_API_TOKEN` | HuggingFace API token (used for embeddings and LLM inference) |
| `K`                        | Number of top-k similar chunks to retrieve (default: `5`)     |

---

## 2. Document Ingestion & Chunking

The ingestion pipeline lives in `ingestion_pipeline.ipynb`. Run the notebook cells sequentially to process PDF documents.

### Pipeline Steps

1. **Partition** – PDFs are parsed with `unstructured.partition.pdf` using the `hi_res` strategy with table structure inference and image extraction.
2. **Chunk** – Elements are grouped into semantically coherent chunks using `chunk_by_title`.
3. **AI Summarisation** – Chunks containing tables or images are enriched with AI-generated searchable summaries via the `moonshotai/Kimi-K2.6` model.
4. **Embed & Store** – Each chunk is embedded with `Qwen/Qwen3-Embedding-8B` (via Scaleway inference provider) and stored in ChromaDB.

### Chunking Strategy

| Parameter                    | Value | Rationale                                                                                                       |
| ---------------------------- | ----- | --------------------------------------------------------------------------------------------------------------- |
| `max_characters`             | 3 000 | Upper bound per chunk; keeps context within embedding model limits                                              |
| `new_after_n_chars`          | 2 400 | Soft split target; encourages chunks of ~2 400 chars before forcing a break                                     |
| `combine_text_under_n_chars` | 500   | Merges very small elements (captions, short paragraphs) into the preceding chunk to avoid under-sized fragments |
| `overlap`                    | 0     | No character overlap between chunks (default)                                                                   |
| `overlap_all`                | False | Overlap is not applied across non-consecutive chunks (default)                                                  |

The `chunk_by_title` strategy splits on document section titles, preserving logical section boundaries rather than splitting at arbitrary character offsets. This produces chunks that are topically coherent and improve retrieval precision.

Character overlap (as used by offset-based splitters like `RecursiveCharacterTextSplitter`) is intentionally omitted here. Because splits occur at semantic boundaries (section headings), context is naturally preserved at each boundary without needing to duplicate text across adjacent chunks.

### Running Ingestion

```python
# Inside the notebook – process a single PDF:
run_complete_ingestion_pipeline("./docs/diabetes_6.pdf")
```

To process all PDFs at once, uncomment the batch processing cell that uses `ThreadPoolExecutor`.

---

## 3. Vector Store Setup

### Database

- **Engine**: ChromaDB (via `langchain-chroma`)
- **Embedding model**: `Qwen/Qwen3-Embedding-8B` (hosted on Scaleway via HuggingFace Inference API)
- **Distance metric**: Cosine similarity (`hnsw:space = cosine`)
- **Persistence**: The vector store is persisted to `db/chroma_db/`. Once built, it is loaded automatically at query time — no rebuild needed.

### Rebuilding the Vector Store

If you need to rebuild from scratch:

1. Delete the `db/chroma_db/` directory.
2. Place your PDF documents in `docs/`.
3. Run the ingestion notebook (`ingestion_pipeline.ipynb`) end-to-end.

---

## 4. API – FastAPI `/ask` Endpoint

### Start the Server

```bash
python deploy.py
```

The server starts on `http://0.0.0.0:8000`.

### Endpoints

#### `POST /ask`

Submit a question and receive an answer grounded in the ingested documents.

**Request:**

```json
{
  "query": "What are the symptoms of type 2 diabetes?"
}
```

**Response:**

```json
{
  "answer": "Common symptoms include increased thirst, frequent urination, ...",
  "sources": [
    {
      "content": "The chunk text retrieved from the vector store...",
      "metadata": {},
      "similarity_score": 0.8234
    }
  ]
}
```

| Field                        | Description                                            |
| ---------------------------- | ------------------------------------------------------ |
| `answer`                     | LLM-generated answer based on retrieved context        |
| `sources[].content`          | Text of the retrieved source chunk                     |
| `sources[].metadata`         | Chunk metadata (original content, tables, images)      |
| `sources[].similarity_score` | Cosine similarity score (0–1, higher is more relevant) |

#### `GET /`

Health-check endpoint.

### Configuring `k` (number of retrieved chunks)

Set the `K` environment variable in `.env` or your shell before starting the server:

```powershell
# PowerShell (Windows)
$env:K = "10"; python deploy.py
```

```bash
# CMD (Windows)
set K=10 && python deploy.py
```

```bash
# Bash / Linux / macOS
K=10 python deploy.py
```

Alternatively, just edit the `K` value directly in your `.env` file.

### Example `curl` Request

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the risk factors for diabetes?"}'
```

---
