🚀 AI Quality Dashboard  https://huvimal-ai-quality-dashboard.hf.space

An advanced RAG Evaluation & Monitoring Dashboard designed to measure, trace, and analyze the quality of AI-generated responses using LLM-as-a-Judge evaluation, hybrid retrieval, and observability tools.

This project demonstrates how to build a lightweight yet production-oriented AI Quality Monitoring System for Retrieval-Augmented Generation (RAG) pipelines.

📌 Overview

AI Quality Dashboard helps developers:

🔍 Evaluate RAG response quality automatically

📊 Monitor retrieval and answer performance

🧠 Use LLMs as automated judges

⚡ Benchmark AI system quality in real time

📈 Track latency and overall response reliability


The project combines:

Hybrid Retrieval (BM25 + Vector Search)

LLM-as-a-Judge Evaluation

LangSmith Tracing

Interactive Dashboard UI


✨ Features

🤖 RAG Pipeline with Hybrid Search

📊 Real-time AI Quality Dashboard

🧠 LLM-as-a-Judge evaluation system

🔍 Metrics:

Faithfulness

Relevancy

Conciseness

Overall Quality

📈 Query analytics & dashboard statistics

⚡ Latency tracking

🧩 LangSmith tracing integration

🌐 Interactive Gradio interface

🏗️ Project Structure

.
├── app.py              # Gradio dashboard UI
├── rag_pipeline.py     # Hybrid RAG pipeline
├── evaluator.py        # LLM-as-judge evaluation
├── requirements.txt    # Dependencies
└── .env


⚙️ Tech Stack

Framework: Gradio

LLM: Groq (Qwen/Qwen3-32B)

AI Frameworks: LangChain, LangGraph

Vector Store: ChromaDB

Retrieval: BM25 + Vector Search

Observability: LangSmith

Embeddings: HuggingFace Embeddings

🧠 System Architecture

🔍 Retrieval Pipeline

1. BM25 Retriever

Keyword-based retrieval

Strong lexical matching

2. Vector Retriever

Semantic similarity search

ChromaDB embeddings

3. Ensemble Fusion

Weighted combination:

BM25 → 0.4

Vector Search → 0.6

📊 Dashboard Features
Query Evaluation

Ask questions directly

View retrieved context

Analyze evaluation scores

Analytics Dashboard

Average quality metrics

Query success rate

Latency tracking

Overall system quality


🚀 Getting Started

1. Clone repository

git clone https://github.com/huvimal/Ai-Quality-Dashboard.git

cd Ai-Quality-Dashboard

2. Install dependencies

pip install -r requirements.txt

3. Setup environment variables

GROQ_API_KEY=your_api_key

LANGCHAIN_API_KEY=your_langsmith_key

4. Run application

python app.py

Application runs at:

http://127.0.0.1:7860

📡 Example Questions

LangSmith dùng để làm gì?
RAGAS đo lường những gì?
Prompt injection là gì?
Helicone giúp tiết kiệm chi phí thế nào?

🎯 Use Cases

✅ RAG evaluation

✅ AI observability

✅ LLM benchmarking

✅ Prompt evaluation

✅ Retrieval quality analysis

✅ AI system monitoring

📈 Future Improvements

🔥 RAGAS integration

📊 Historical analytics database

🧠 Multi-model evaluation

📁 PDF document ingestion

🌐 Production monitoring dashboard

⚡ Semantic caching

🔐 Authentication & user analytics

👨‍💻 Author

Huvimal
