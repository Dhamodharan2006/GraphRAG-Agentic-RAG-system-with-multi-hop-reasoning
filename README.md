# GraphRAG-Agentic-RAG-system-with-multi-hop-reasoning
GraphRAG + Agentic RAG system with multi-hop reasoning over 30–50 research papers by building a knowledge graph from unstructured data

📌 Overview

This project implements a GraphRAG + Agentic RAG system designed to perform multi-hop reasoning across 30–50 research papers. It combines knowledge graphs (Neo4j) with vector-based retrieval (ChromaDB + Hugging Face embeddings) to generate accurate, explainable answers from unstructured academic data.

The system leverages Groq API (fast inference) and Google Gemini API (advanced reasoning) to deliver scalable and efficient LLM-powered responses.

🧠 Key Features
🔗 Knowledge Graph Construction from unstructured research papers
🔍 Hybrid Retrieval (Vector + Graph-based reasoning)
🔄 Multi-Hop Reasoning across interconnected entities
⚡ Fast Inference using Groq API
🤖 Advanced Reasoning via Gemini API
📊 Explainable AI Outputs with reasoning paths
🛠️ Technical Architecture
User Query
   ↓
Entity Extraction (LLM / spaCy)
   ↓
Graph Retrieval (Neo4j) + Vector Retrieval (ChromaDB)
   ↓
Multi-Hop Reasoning Engine
   ↓
LLM Response Generation (Groq / Gemini)
⚙️ Core Components
1. Entity & Relationship Extraction
Replaced naive bag-of-words extraction with:
✅ LLM-based NER
✅ spaCy-based entity recognition
Ensures accurate graph queries and meaningful reasoning
2. Knowledge Graph (Neo4j)
Stores:
Entities (papers, methods, models)
Relationships (comparisons, improvements, dependencies)
Removed APOC dependency
Replaced with pure Cypher queries
Ensures production compatibility
3. Vector Store (ChromaDB)
Uses Hugging Face embeddings
Fixed issue of dual embedding mismatch
✅ Unified embedding model across graph & vector DB
Enables true hybrid semantic search
4. Multi-Hop Reasoning Engine
Traverses graph relationships
Combines results with vector similarity
Generates context-aware answers across documents
