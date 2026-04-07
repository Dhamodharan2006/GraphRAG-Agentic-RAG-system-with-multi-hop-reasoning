import json
import re
from agents.state import AgentState
from agents.tools import tool_vector_search, tool_multihop_paths
from core.groq_client import get_client
from core.config import GROQ_MODEL


def node_query_planner(state: AgentState) -> AgentState:
    """Analyzes question and decides retrieval strategy"""
    question = state.question.lower()
    client = get_client()

    planning_prompt = f"""Analyze this research question and decide the best retrieval strategy.

Question: "{state.question}"

Available strategies:
- "vector_only": Simple factual lookup (what is X?, define Y)
- "graph_only": Relationship/path questions (how does X relate to Y?)
- "hybrid": Comparison or synthesis questions (compare X and Y, analyze, summarize)

Respond with ONLY a JSON object:
{{"strategy": "vector_only|graph_only|hybrid", "plan": ["step1", "step2"], "reasoning": "brief explanation"}}

Strategy selection rules:
- "what is", "define", "explain", "summarize", "each paper", "overview" → vector_only
- "how does", "relate", "connect", "path between" → graph_only
- "compare", "vs", "difference", "trade-off", "best", "which" → hybrid"""

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": planning_prompt}],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        plan_data = json.loads(response.choices[0].message.content)
        state.strategy = plan_data.get('strategy', 'hybrid')
        # IMPORTANT: always include vector_search in plan so retriever never skips
        raw_plan = plan_data.get('plan', [])
        if 'vector_search' not in raw_plan:
            raw_plan = ['vector_search'] + raw_plan
        state.plan = raw_plan

    except Exception as e:
        print(f"Planner LLM error (using fallback): {e}")
        if any(w in question for w in ['compare', 'vs', 'difference', 'better', 'which']):
            state.strategy = "hybrid"
            state.plan = ["vector_search", "graph_explore", "synthesize"]
        elif any(w in question for w in ['how', 'relate', 'connect', 'path between']):
            state.strategy = "graph_only"
            state.plan = ["vector_search", "graph_explore", "synthesize"]
        else:
            state.strategy = "vector_only"
            state.plan = ["vector_search", "synthesize"]

    state.add_trajectory(
        node="QueryPlanner",
        action=f"Selected '{state.strategy}' strategy",
        input_s=state.question,
        output_s=str(state.plan)
    )
    print(f"🎯 Strategy: {state.strategy} | Plan: {state.plan}")
    return state


def node_context_retriever(state: AgentState) -> AgentState:
    """Retrieves relevant chunks from ChromaDB — always runs unless plan explicitly excludes it"""
    # Only skip if graph_only AND vector_search not in plan
    if state.strategy == "graph_only" and "vector_search" not in state.plan:
        state.add_trajectory("ContextRetriever", "skipped — graph_only strategy", state.question, "no action")
        return state

    results = tool_vector_search(state.question, top_k=5)
    state.vector_context = results
    state.token_usage += sum(len(r['text']) for r in results) // 4

    papers_found = list(set(r['paper_id'] for r in results))
    state.add_trajectory(
        node="ContextRetriever",
        action=f"Retrieved {len(results)} chunks from {len(papers_found)} paper(s)",
        input_s=state.question,
        output_s=str(papers_found)
    )
    print(f"📚 Retrieved {len(results)} chunks from: {papers_found}")
    return state


def node_graph_explorer(state: AgentState) -> AgentState:
    """Explores knowledge graph for multi-hop reasoning"""
    if state.strategy == "vector_only":
        state.add_trajectory("GraphExplorer", "skipped — vector_only strategy", "", "no action")
        return state

    stopwords = {'what', 'which', 'how', 'does', 'and', 'for', 'the', 'with',
                 'are', 'from', 'that', 'this', 'each', 'paper', 'about',
                 'explain', 'give', 'tell', 'describe'}
    potential_entities = [
        w.strip('?.,') for w in state.question.split()
        if len(w) > 3 and w.lower().strip('?.,') not in stopwords
    ]

    all_paths = []

    if len(potential_entities) >= 2:
        for i in range(min(3, len(potential_entities))):
            for j in range(i + 1, min(4, len(potential_entities))):
                paths = tool_multihop_paths(potential_entities[i], potential_entities[j], max_hops=3)
                all_paths.extend(paths)

    if not all_paths and potential_entities:
        for entity in potential_entities[:3]:
            paths = tool_multihop_paths(entity, max_hops=3)
            all_paths.extend(paths)

    seen = set()
    unique_paths = []
    for path in all_paths:
        key = tuple(path.nodes)
        if key not in seen:
            seen.add(key)
            unique_paths.append(path)

    state.graph_paths = unique_paths[:5]
    state.add_trajectory(
        node="GraphExplorer",
        action=f"Found {len(unique_paths)} unique graph paths",
        input_s=str(potential_entities[:4]),
        output_s=str([p.nodes for p in unique_paths[:2]])
    )
    print(f"🕸️ Found {len(unique_paths)} graph paths")
    return state


def node_synthesizer(state: AgentState) -> AgentState:
    """Combines vector + graph context into final answer"""
    client = get_client()

    vector_section = ""
    if state.vector_context:
        vector_section = "RELEVANT TEXT CHUNKS FROM DOCUMENTS:\n"
        for i, chunk in enumerate(state.vector_context, 1):
            vector_section += (
                f"[{i}] From paper '{chunk.get('paper_id', 'unknown')}' "
                f"(section: {chunk.get('section', 'content')}):\n"
                f"{chunk['text'][:400]}\n\n"
            )
    else:
        vector_section = "No document chunks were retrieved.\n"

    graph_section = ""
    if state.graph_paths:
        graph_section = "KNOWLEDGE GRAPH PATHS:\n"
        for i, path in enumerate(state.graph_paths, 1):
            if path.nodes and path.relationships:
                path_str = " → ".join(
                    [f"{n} [{r}]" for n, r in zip(path.nodes[:-1], path.relationships)]
                ) + f" → {path.nodes[-1]}"
                graph_section += f"[{i}] {path_str}\n"

    synthesis_prompt = f"""You are a research assistant. Answer the question using ONLY the context provided below.

Question: {state.question}

{vector_section}
{graph_section}

Instructions:
1. Answer based on the document chunks provided above
2. Cite sources using [1], [2] etc. referencing the chunk numbers
3. If comparing approaches, highlight trade-offs explicitly
4. If no relevant chunks exist, say so honestly
5. End with: "Confidence: high/medium/low"

Answer:"""

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": synthesis_prompt}],
            temperature=0.3,
            max_tokens=1000
        )
        state.synthesis = response.choices[0].message.content

        text_lower = state.synthesis.lower()
        if "high confidence" in text_lower or "confidence: high" in text_lower:
            state.confidence = 0.9
        elif "medium confidence" in text_lower or "confidence: medium" in text_lower:
            state.confidence = 0.7
        elif "low confidence" in text_lower or "confidence: low" in text_lower:
            state.confidence = 0.5
        else:
            state.confidence = 0.75

        state.citations = list(set(re.findall(r'\[(\d+)\]', state.synthesis)))

    except Exception as e:
        state.synthesis = f"Error generating answer: {e}"
        state.confidence = 0.0

    state.add_trajectory(
        node="Synthesizer",
        action=f"Generated answer ({len(state.synthesis)} chars)",
        input_s=f"{len(state.vector_context)} chunks, {len(state.graph_paths)} graph paths",
        output_s=f"Confidence: {state.confidence}"
    )
    print(f"✅ Synthesized answer (confidence: {state.confidence})")
    return state