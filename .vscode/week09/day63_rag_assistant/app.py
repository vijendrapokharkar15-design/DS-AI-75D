import os
import gradio as gr
import chromadb
import anthropic
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import urllib.request
import re
from gradio import ChatMessage

embedder = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.Client()
collection = chroma_client.create_collection("rag_docs")
anthropic_client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

def chunk_text(text, chunk_size=200, overlap=20):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return chunks

def ingest_pdf(pdf_file):
    try:
        reader = PdfReader(pdf_file.name)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + " "
        chunks = chunk_text(text)
        embeddings = embedder.encode(chunks).tolist()
        ids = [f"pdf_chunk_{i}" for i in range(len(chunks))]
        collection.add(documents=chunks, embeddings=embeddings, ids=ids,
                      metadatas=[{"source": pdf_file.name, "type": "pdf"} for _ in chunks])
        return f"✅ PDF ingested: {len(chunks)} chunks added."
    except Exception as e:
        return f"❌ Error: {str(e)}"

def ingest_url(url):
    try:
        with urllib.request.urlopen(url) as response:
            text = response.read().decode("utf-8", errors="ignore")
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        chunks = chunk_text(text)
        embeddings = embedder.encode(chunks).tolist()
        ids = [f"url_chunk_{i}" for i in range(len(chunks))]
        collection.add(documents=chunks, embeddings=embeddings, ids=ids,
                      metadatas=[{"source": url, "type": "url"} for _ in chunks])
        return f"✅ URL ingested: {len(chunks)} chunks added."
    except Exception as e:
        return f"❌ Error: {str(e)}"

def retrieve(query, n=3):
    q_emb = embedder.encode([query]).tolist()
    results = collection.query(query_embeddings=q_emb,
                               n_results=min(n, collection.count()),
                               include=["documents", "metadatas", "distances"])
    return [{"text": results["documents"][0][i],
             "source": results["metadatas"][0][i]["source"],
             "distance": round(results["distances"][0][i], 4)}
            for i in range(len(results["ids"][0]))]

def answer_with_citations(question, history):
    history = history or []

    if collection.count() == 0:
        yield history + [
            ChatMessage(role="user", content=question),
            ChatMessage(role="assistant", content="⚠️ No documents ingested yet. Please add a PDF or URL first.")
        ]
        return

    chunks = retrieve(question)
    context = "\n\n".join([f"[{i+1}] (Source: {c['source'][:50]})\n{c['text']}"
                           for i, c in enumerate(chunks)])
    prompt = f"""Answer using ONLY the context below. Add citations [1][2] after claims.
If not in context, say 'I don't have enough information.'

Context:
{context}

Question: {question}"""

    sources = list(set([c["source"] for c in chunks]))
    sources_text = "\n\n**Sources:**\n" + "\n".join([f"- {s}" for s in sources])

    new_history = history + [
        ChatMessage(role="user", content=question),
        ChatMessage(role="assistant", content="")
    ]

    response_text = ""
    with anthropic_client.messages.stream(
        model="claude-haiku-4-5", max_tokens=500,
        messages=[{"role": "user", "content": prompt}]
    ) as stream:
        for text in stream.text_stream:
            response_text += text
            new_history[-1] = ChatMessage(role="assistant", content=response_text)
            yield new_history

    new_history[-1] = ChatMessage(role="assistant", content=response_text + sources_text)
    yield new_history

with gr.Blocks(title="RAG Assistant") as demo:
    gr.Markdown("# 🔍 AI-Powered RAG Assistant\nIngest documents and ask questions with source citations.")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📄 Add Documents")
            pdf_input = gr.File(label="Upload PDF", file_types=[".pdf"])
            pdf_btn = gr.Button("Ingest PDF", variant="secondary")
            pdf_status = gr.Textbox(label="PDF Status", interactive=False)
            gr.Markdown("---")
            url_input = gr.Textbox(label="Or enter a URL")
            url_btn = gr.Button("Ingest URL", variant="secondary")
            url_status = gr.Textbox(label="URL Status", interactive=False)

        with gr.Column(scale=2):
            gr.Markdown("### 💬 Ask Questions")
            chatbot = gr.Chatbot(height=400, label="RAG Assistant")
            question_input = gr.Textbox(
                label="Your question",
                placeholder="What does the document say about...?",
                lines=2
            )
            ask_btn = gr.Button("Ask", variant="primary")
            clear_btn = gr.Button("Clear Chat")

    pdf_btn.click(ingest_pdf, inputs=[pdf_input], outputs=[pdf_status])
    url_btn.click(ingest_url, inputs=[url_input], outputs=[url_status])
    ask_btn.click(answer_with_citations, inputs=[question_input, chatbot], outputs=[chatbot])
    question_input.submit(answer_with_citations, inputs=[question_input, chatbot], outputs=[chatbot])
    clear_btn.click(lambda: [], outputs=[chatbot])

if __name__ == "__main__":
   demo.launch(server_name="0.0.0.0", server_port=7860)