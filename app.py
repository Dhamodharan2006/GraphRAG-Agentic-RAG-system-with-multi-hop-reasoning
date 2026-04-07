import chainlit as cl
import os
import asyncio
from core.config import UPLOAD_PATH
from core.pdf_processor import extract_chunks
from core.vector_db import get_collection
from core.groq_client import groq_chat
from agents.workflow import agent_app
from agents.state import AgentState
from rag.normal_rag import normal_rag_answer

os.makedirs(UPLOAD_PATH, exist_ok=True)


# ── Mode Toggle in Chat Bar ───────────────────────────────────────
@cl.set_chat_profiles
async def chat_profile():
    return [
        cl.ChatProfile(
            name="NormalRAG",
            markdown_description="**🔍 Normal RAG** — Fast vector similarity search over your documents.",
            icon="https://cdn-icons-png.flaticon.com/512/2092/2092663.png",
        ),
        cl.ChatProfile(
            name="GraphRAG",
            markdown_description="**🧠 GraphRAG** — Multi-hop knowledge graph reasoning over your documents.",
            icon="https://cdn-icons-png.flaticon.com/512/2534/2534230.png",
        ),
    ]


# ── Chat Start ────────────────────────────────────────────────────
@cl.on_chat_start
async def on_chat_start():
    profile = cl.user_session.get("chat_profile") or "GraphRAG"
    cl.user_session.set("rag_mode", profile)
    cl.user_session.set("uploaded_docs", [])

    icon = "🔍" if profile == "NormalRAG" else "🧠"
    await cl.Message(
        content=(
            f"👋 **Welcome to GraphRAG Research Assistant!**\n\n"
            f"Active mode: **{icon} {profile}**\n\n"
            f"Switch modes using the **profile selector** at the top of the chat.\n\n"
            f"- 💬 Type anything to **chat normally** (no PDF needed)\n"
            f"- 📎 **Upload PDFs** then ask questions about them"
        )
    ).send()


# ── Main Message Handler ──────────────────────────────────────────
@cl.on_message
async def on_message(message: cl.Message):
    # Always sync mode from the profile selector
    profile = cl.user_session.get("chat_profile") or "GraphRAG"
    cl.user_session.set("rag_mode", profile)

    uploaded_files = []
    if message.elements:
        for element in message.elements:
            if hasattr(element, 'name') and element.name and element.name.lower().endswith(".pdf"):
                uploaded_files.append(element)

    # Index any uploaded PDFs first
    for f in uploaded_files:
        await handle_pdf_upload(f)

    query = (message.content or "").strip()

    if not query and not uploaded_files:
        await cl.Message(content="Please type a message or attach a PDF 📎").send()
        return

    if query:
        docs = cl.user_session.get("uploaded_docs", [])
        if not docs:
            # No docs yet → plain LLM chat
            await handle_general_chat(query)
        else:
            # Docs indexed → RAG query
            await handle_rag_query(query)


# ── General Chat (no documents) ───────────────────────────────────
async def handle_general_chat(query: str):
    msg = cl.Message(content="💬 Thinking...")
    await msg.send()
    try:
        response = await asyncio.to_thread(
            groq_chat, query,
            "You are a helpful, knowledgeable AI assistant. Answer clearly and concisely."
        )
        msg.content = response
    except Exception as e:
        msg.content = f"❌ Error: `{str(e)}`"
    await msg.update()


# ── PDF Upload & Indexing ─────────────────────────────────────────
async def handle_pdf_upload(file):
    save_path = os.path.join(UPLOAD_PATH, file.name)

    try:
        if hasattr(file, 'content') and file.content:
            file_bytes = file.content
        elif hasattr(file, 'path') and file.path:
            with open(file.path, 'rb') as f:
                file_bytes = f.read()
        else:
            await cl.Message(content=f"❌ Cannot read file: `{file.name}`").send()
            return
    except Exception as e:
        await cl.Message(content=f"❌ File read error: `{str(e)}`").send()
        return

    with open(save_path, "wb") as f:
        f.write(file_bytes)

    msg = cl.Message(content=f"⏳ Processing **{file.name}**...")
    await msg.send()

    try:
        paper_id, title, chunks = await asyncio.to_thread(extract_chunks, save_path)
        col = get_collection()

        try:
            existing = col.get(ids=[f"{paper_id}_chunk_0"])
            already_exists = bool(existing and existing.get('ids'))
        except Exception:
            already_exists = False

        if already_exists:
            msg.content = f"ℹ️ **{file.name}** already indexed (`{len(chunks)}` chunks)"
        else:
            col.add(
                documents=chunks,
                ids=[f"{paper_id}_chunk_{i}" for i in range(len(chunks))],
                metadatas=[{
                    "paper_id": paper_id,
                    "title": title,
                    "section": "content",
                    "chunk_index": i
                } for i in range(len(chunks))]
            )
            msg.content = (
                f"✅ **{file.name}** indexed!\n"
                f"📝 Title: `{title or paper_id}`\n"
                f"🔢 Chunks: `{len(chunks)}`"
            )
        await msg.update()

        docs = cl.user_session.get("uploaded_docs", [])
        if not any(d["paper_id"] == paper_id for d in docs):
            docs.append({
                "name": file.name,
                "paper_id": paper_id,
                "title": title or paper_id,
                "chunks": len(chunks)
            })
            cl.user_session.set("uploaded_docs", docs)
            await update_side_panel(docs)

    except Exception as e:
        msg.content = f"❌ Failed to process **{file.name}**: `{str(e)}`"
        await msg.update()


# ── RAG Query ─────────────────────────────────────────────────────
async def handle_rag_query(question: str):
    mode = cl.user_session.get("rag_mode", "GraphRAG")
    icon = "🔍" if mode == "NormalRAG" else "🧠"

    msg = cl.Message(content=f"{icon} **{mode}** is thinking...")
    await msg.send()

    try:
        if mode == "NormalRAG":
            answer = await asyncio.to_thread(normal_rag_answer, question)
            msg.content = f"**{icon} Answer (Normal RAG):**\n\n{answer}"

        else:
            # Force plan to always include both steps so retriever is never skipped
            state = AgentState(question=question)
            state.plan = ["vector_search", "graph_explore", "synthesize"]

            result = await asyncio.to_thread(agent_app.invoke, state)

            if isinstance(result, dict):
                answer     = result.get("synthesis", "No answer generated.")
                confidence = result.get("confidence", 0.75)
                trajectory = result.get("trajectory", [])
                strategy   = result.get("strategy", "hybrid")
            else:
                answer     = getattr(result, "synthesis", "No answer generated.")
                confidence = getattr(result, "confidence", 0.75)
                trajectory = getattr(result, "trajectory", [])
                strategy   = getattr(result, "strategy", "hybrid")

            emoji_map = {
                "QueryPlanner":     "🎯",
                "ContextRetriever": "📚",
                "GraphExplorer":    "🕸️",
                "Synthesizer":      "📝"
            }
            step_lines = []
            for t in trajectory:
                node   = t.node_name if hasattr(t, "node_name") else t.get("node", "")
                action = t.action    if hasattr(t, "action")    else t.get("action", "")
                step_lines.append(f"{emoji_map.get(node, '➡️')} **{node}**: {action}")

            msg.content = (
                f"**🧠 Answer (GraphRAG — {strategy}):**\n\n"
                f"{answer}\n\n"
                f"---\n"
                f"🎯 Confidence: `{confidence:.0%}` &nbsp;|&nbsp; Strategy: `{strategy}`\n\n"
                f"**Reasoning trace:**\n" + "\n".join(step_lines)
            )

    except Exception as e:
        msg.content = f"❌ Error: `{str(e)}`"

    await msg.update()


# ── Side Panel ────────────────────────────────────────────────────
async def update_side_panel(docs: list):
    if not docs:
        return
    elements = [
        cl.Text(
            name=doc["name"],
            content=(
                f"📄 **{doc['title']}**\n\n"
                f"- File: `{doc['name']}`\n"
                f"- Chunks: `{doc['chunks']}`\n"
                f"- ID: `{doc['paper_id']}`"
            ),
            display="side"
        )
        for doc in docs
    ]
    await cl.Message(
        content=f"📚 **Library — {len(docs)} document(s) ready**",
        elements=elements
    ).send()