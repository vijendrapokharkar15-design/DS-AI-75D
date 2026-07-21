# 📰 BBC News Classifier

Classifies news articles into one of five categories — business, entertainment, politics, sport, tech — using a fine-tuned DistilBERT model served through a FastAPI endpoint. Built as Project 2 of my 75-day DS/AI roadmap.

🌐 **Live App:** [Hugging Face Space](https://huggingface.co/spaces/VijendraHuggingface/bbc-news-classifier)

---

## Problem

Classic NLP classification problems are usually solved and left as a notebook metric. The goal here was to go further — fine-tune a real transformer model, then actually serve it behind a production-style API with request validation and auto-generated docs, then containerize and deploy it, the way it'd be shipped at a company.

## Approach

- Fine-tuned DistilBERT on the BBC News dataset (5 categories: business, entertainment, politics, sport, tech) using the Hugging Face Trainer API, 2 epochs.
- Built a FastAPI `/predict` endpoint with Pydantic request/response validation and auto-generated Swagger docs.
- Dockerized the service and deployed it to Hugging Face Spaces.

## Results

**Overall:**

| Metric | Value |
|---|---|
| Validation accuracy | 98.65% |
| Test accuracy | 99% |

**Per-category (test set, 445 samples):**

| Category | Precision | Recall | F1-score |
|---|---|---|---|
| Business | 0.99 | 0.95 | 0.97 |
| Entertainment | 0.99 | 1.00 | 0.99 |
| Politics | 0.97 | 0.99 | 0.98 |
| Sport | 1.00 | 1.00 | 1.00 |
| Tech | 0.99 | 1.00 | 0.99 |

## Tech Stack

`Python` `PyTorch` `Hugging Face Transformers` `DistilBERT` `FastAPI` `Pydantic` `Docker`

## How to Run Locally

**Option 1 — Directly with uvicorn:**
```bash
git clone https://github.com/vijendrapokharkar15-design/DS-AI-75D.git
cd DS-AI-75D/.vscode/week07/day49_fastapi_app
pip install -r requirements.txt
uvicorn main:app --reload
```

**Option 2 — With Docker:**
```bash
cd DS-AI-75D/.vscode/week07/day49_fastapi_app
docker build -t bbc-news-classifier .
docker run -p 7860:7860 bbc-news-classifier
```

Then visit `http://localhost:7860/docs` for the interactive Swagger UI.

## Part of DS-AI-75D Journey

This project is part of my [75-Day Data Science & AI Roadmap](https://github.com/vijendrapokharkar15-design/DS-AI-75D) — built on Days 48–49 (Phase 3).