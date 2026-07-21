# 🔍 AI-Powered RAG Assistant

A document Q&A assistant that ingests PDFs and URLs, retrieves relevant chunks via semantic search, and answers questions with inline source citations — powered by Claude with streaming responses. Built as Project 3 of my 75-day DS/AI roadmap.

🌐 **Live App:** [Hugging Face Space](https://huggingface.co/spaces/VijendraHuggingface/rag-assistant)
> Note: the Space sleeps after inactivity (free tier) — first load may take 30–60 seconds to wake up.

---

## Problem

A RAG system is only actually useful if you can trust its answers. The goal here wasn't just "retrieve and generate" — it was to make every claim traceable back to a source, so a user can verify the assistant isn't making things up.

## Approach

- Documents (PDF or URL) are chunked into ~200-word segments with 20-word overlap, embedded with `all-MiniLM-L6-v2`, and stored in a ChromaDB collection.
- At query time, the top-3 most relevant chunks are retrieved by semantic similarity.
- Claude (Haiku 4.5) generates an answer constrained to only use the retrieved context, with inline `[1][2]` citation markers, streamed token-by-token into the chat UI.
- Every answer ends with a "Sources" list showing exactly which documents were used.
- If no documents have been ingested yet, the assistant says so instead of guessing.

## Features

- 📄 **PDF ingestion** — upload and index any PDF
- 🔗 **URL ingestion** — pull and index text from any webpage
- 💬 **Chat interface** — multi-turn conversation with streaming responses
- 📌 **Source citations** — every answer traces back to its source chunk(s)

## Tech Stack

`Python` `Gradio` `ChromaDB` `sentence-transformers` `Anthropic API (Claude Haiku 4.5)` `pypdf`

## How to Run Locally

```bash
git clone https://github.com/vijendrapokharkar15-design/DS-AI-75D.git
cd DS-AI-75D/.vscode/week09/day63_rag_assistant
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your_key_here
python app.py
```
Then visit `http://localhost:7860`.

## Current Limitations

- Retrieval is semantic-only (no hybrid/BM25 or reranking yet)
- ChromaDB runs in-memory — ingested documents don't persist across restarts

## Part of DS-AI-75D Journey

This project is part of my [75-Day Data Science & AI Roadmap](https://github.com/vijendrapokharkar15-design/DS-AI-75D) — built on Day 63 (Phase 4).