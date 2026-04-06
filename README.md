# 🚀 GraphRAG + Agentic RAG System with Multi-Hop Reasoning

A scalable **GraphRAG + Agentic RAG system** designed to perform **multi-hop reasoning across 30–50 research papers** by constructing a **knowledge graph from unstructured data**.

---

## 📌 Overview

This project combines:

- 🧠 **Knowledge Graphs (Neo4j)**
- 🔍 **Vector Retrieval (ChromaDB + Hugging Face embeddings)**
- ⚡ **Fast LLM Inference (Groq API)**
- 🤖 **Advanced Reasoning (Google Gemini API)**

to generate **accurate, explainable, and context-aware answers** from academic research data.

---

## 🧠 Key Features

- 🔗 **Knowledge Graph Construction** from unstructured research papers  
- 🔍 **Hybrid Retrieval** (Vector + Graph-based reasoning)  
- 🔄 **Multi-Hop Reasoning** across interconnected entities  
- ⚡ **Fast Inference** using Groq API  
- 🤖 **Advanced Reasoning** via Gemini API  
- 📊 **Explainable Outputs** with reasoning paths  

---

## 🛠️ Technical Architecture

```text
User Query
   ↓
Entity Extraction (LLM / spaCy)
   ↓
Graph Retrieval (Neo4j) + Vector Retrieval (ChromaDB)
   ↓
Multi-Hop Reasoning Engine
   ↓
LLM Response Generation (Groq / Gemini)
