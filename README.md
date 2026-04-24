# Enhanced Document Q&A System with Intelligent RAG Pipeline

A production-ready document question-answering system that processes multi-document PDFs using an intelligent RAG (Retrieval-Augmented Generation) pipeline. The system automatically detects document boundaries, classifies document types, and routes queries to the most relevant content — all through a user-friendly Gradio interface.

---

## Features

- **Intelligent Multi-Document Detection** — Automatically identifies and separates distinct documents within a single PDF (e.g., a packet containing a resume, contract, and invoice)
- **LLM-Based Document Classification** — Classifies each document into one of 16 categories (Resume, Contract, Invoice, Pay Slip, Bank Statement, etc.) using Google Gemini
- **Query Routing** — Predicts which document type is most likely to contain the answer and routes retrieval accordingly for improved accuracy
- **Rich Metadata Preservation** — Tracks document type, page ranges, and chunk provenance throughout the pipeline
- **Dual Chunking Strategies** — Supports both a custom sliding-window chunker and LlamaIndex's `SentenceSplitter` for flexible chunking
- **FAISS Vector Search** — Builds both a global index and per-document-type sub-indices for fast, filtered retrieval
- **Source Attribution** — Every answer includes the source document type, page range, and relevance score
- **OCR Fallback** — Falls back to Tesseract OCR for scanned or image-based PDF pages
- **Gradio Interface** — A clean, single-tab UI with inline PDF preview, settings panel, and chat

---

## Tech Stack

| Component | Library |
|---|---|
| PDF Extraction | PyMuPDF (`fitz`), PyPDF2 |
| OCR | Tesseract via `pytesseract` |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) |
| Vector Store | FAISS (`faiss-cpu`) |
| LLM | Google Gemini (`gemini-2.0-flash`) |
| Advanced Chunking | LlamaIndex (`SentenceSplitter`) |
| UI | Gradio + `gradio_pdf` |

---

## Installation

```bash
pip install gradio gradio_pdf
pip install pypdf PyPDF2 pymupdf
pip install sentence-transformers transformers
pip install faiss-cpu
pip install google-generativeai
pip install numpy pandas
pip install llama-index llama-index-readers-file
pip install llama-index-embeddings-huggingface
pip install llama-index-vector-stores-faiss
pip install llama-index-llms-gemini
```

---

## Configuration

Before running, set your Gemini API key in the notebook:

```python
GEMINI_API_KEY = "your-api-key-here"
genai.configure(api_key=GEMINI_API_KEY)
```

Get a key at [https://aistudio.google.com](https://aistudio.google.com).

---

## Usage

1. Open the notebook and run all cells.
2. The final cell launches the Gradio app:
   ```python
   demo = create_interface()
   demo.launch(share=True, debug=True)
   ```
3. Upload a PDF in the viewer. The system will automatically:
   - Extract text from each page (with OCR fallback for scanned pages)
   - Detect document boundaries between consecutive pages
   - Classify each logical document by type
   - Chunk documents and build FAISS indices
4. Ask questions in the chat panel. Use the **Document Type Filter** dropdown to restrict search to a specific document, or leave it on **All** with **Auto-Route Queries** enabled to let the system decide.

---

## Pipeline Architecture

```
PDF Upload
    │
    ▼
Page Text Extraction (PyMuPDF + OCR fallback)
    │
    ▼
Document Boundary Detection (Gemini LLM, page-by-page)
    │
    ▼
Document Type Classification (Gemini LLM)
    │
    ▼
Chunking with Metadata (custom or LlamaIndex SentenceSplitter)
    │
    ▼
Embedding (all-MiniLM-L6-v2)
    │
    ├── Global FAISS Index
    └── Per-Document-Type FAISS Sub-Indices
                │
                ▼
         Query Routing (Gemini predicts target doc type)
                │
                ▼
         Retrieval (filtered or global search)
                │
                ▼
         Answer Generation with Source Attribution (Gemini)
```

---

## Key Classes & Functions

| Name | Description |
|---|---|
| `EnhancedDocumentStore` | Top-level orchestrator for the full processing and query pipeline |
| `IntelligentRetriever` | Manages FAISS indices and handles routed/filtered retrieval |
| `classify_document_type()` | Uses Gemini to classify a page's text into one of 16 document categories |
| `detect_document_boundary()` | Uses Gemini to determine whether two consecutive pages belong to the same document |
| `predict_query_document_type()` | Routes a query to the most relevant document type with a confidence score |
| `chunk_document_with_metadata()` | Custom sliding-window chunker with overlap and page-range tracking |
| `chunk_with_llama_index()` | LlamaIndex-based chunker using `SentenceSplitter` |
| `generate_answer_with_sources()` | Generates a grounded answer from retrieved chunks with source citations |

---

## Supported Document Types

Resume, Contract, Mortgage Contract, Invoice, Pay Slip, Lender Fee Sheet, Land Deed, Bank Statement, Tax Document, Insurance, Report, Letter, Form, ID Document, Medical, Other

---

## Notes

- The system makes multiple Gemini API calls per page during ingestion (classification + boundary detection), so processing time scales with document length and API latency.
- For very large PDFs, consider batching or caching classification results to reduce API costs.
- The `share=True` flag in `demo.launch()` creates a public Gradio link — remove it if running in a private environment.
