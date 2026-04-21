"""
agent.py — Study Buddy (Physics) Shared Agent Module
Sanchita Singh | Capstone Project

This module builds and returns the LangGraph agent, embedder, and ChromaDB collection.
Import this in capstone_streamlit.py to keep the UI layer separate from agent logic.
"""

import os
from dotenv import load_dotenv

load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, List
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── Constants ─────────────────────────────────────────────
FAITHFULNESS_THRESHOLD = 0.7
MAX_EVAL_RETRIES       = 2

# ── LLM ───────────────────────────────────────────────────
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

# ── State ─────────────────────────────────────────────────
class CapstoneState(TypedDict):
    question:         str
    messages:         List[dict]
    route:            str
    intent:           str
    retrieved:        str
    sources:          List[str]
    tool_name:        str
    tool_input:       str
    tool_result:      str
    answer:           str
    faithfulness:     float
    eval_retries:     int
    topic:            str
    difficulty_level: str
    formula_used:     str
    steps:            str
    diagram:          bool
    search_results:   str


# ── Knowledge Base Loader ─────────────────────────────────
def build_knowledge_base(pdf_files: list):
    """
    Load PDF files, chunk them, embed with SentenceTransformer,
    and store in ChromaDB. Returns (embedder, collection).
    """
    all_docs = []
    for file in pdf_files:
        if os.path.exists(file):
            loader = PyPDFLoader(file)
            all_docs.extend(loader.load())
        else:
            print(f"⚠️  File not found: {file}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks   = splitter.split_documents(all_docs)

    texts     = []
    metadatas = []
    for doc in chunks:
        source = doc.metadata.get("source", "")
        topic  = os.path.basename(source).replace(".pdf", "")
        texts.append(doc.page_content)
        metadatas.append({"source": source, "topic": topic})

    print("⏳ Loading embedding model…")
    embedder   = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedder.encode(texts, show_progress_bar=True).tolist()

    client = chromadb.Client()
    try:
        client.delete_collection("capstone_kb")
    except Exception:
        pass

    collection = client.create_collection("capstone_kb")
    collection.add(
        documents=texts,
        embeddings=embeddings,
        ids=[f"doc_{i}" for i in range(len(texts))],
        metadatas=metadatas,
    )
    print(f"✅ Knowledge base ready: {collection.count()} chunks")
    return embedder, collection


# ── Node Functions ────────────────────────────────────────

def build_agent(embedder, collection):
    """
    Compile and return the LangGraph agent given an embedder and ChromaDB collection.
    """

    # Node 1: Memory
    def memory_node(state: CapstoneState) -> dict:
        msgs = state.get("messages", [])
        msgs = msgs + [{"role": "user", "content": state["question"]}]
        if len(msgs) > 6:
            msgs = msgs[-6:]
        return {"messages": msgs}

    # Node 2: Router
    def router_node(state: CapstoneState) -> dict:
        question = state["question"]
        messages = state.get("messages", [])

        recent = "; ".join(
            f"{m['role']}: {m['content'][:60]}"
            for m in messages[-3:-1]
        ) or "none"

        prompt = f"""
You are a routing assistant for a Physics Study Buddy chatbot.

Decide the best action to answer the student's question.

Available options:
- retrieve     → for physics concepts from syllabus (SHM, Laws of Motion, Waves, Optics, etc.)
- memory_only  → if question refers to previous conversation (e.g., "what did you just explain?")
- tool         → if special capability is needed

Use tool when:
- question involves numerical problem solving → use solver tool
- question asks for derivation               → use derivation tool
- question asks for study plan               → use plan tool
- question needs visualization/diagram       → use visualizer tool
- question asks for latest/out-of-syllabus info → use web search tool

Recent conversation: {recent}
Current question: {question}

Reply with ONLY ONE WORD: retrieve OR memory_only OR tool
"""
        response = llm.invoke(prompt)
        decision = response.content.strip().lower()

        if "memory" in decision:
            route = "memory_only"
        elif "tool" in decision:
            route = "tool"
        else:
            route = "retrieve"

        # Derive intent
        q = question.lower()
        if any(k in q for k in ["latest", "recent", "new", "current", "2024", "2025"]):
            intent = "search"
        elif any(k in q for k in ["solve", "calculate", "numerical", "find the value", "find the force"]):
            intent = "numerical"
        elif any(k in q for k in ["study plan", "roadmap", "how to prepare", "schedule"]):
            intent = "plan"
        elif any(k in q for k in ["derive", "derivation", "prove the formula"]):
            intent = "derivation"
        else:
            intent = "concept"

        return {"route": route, "intent": intent}

    # Node 3: Retrieval
    def retrieval_node(state: CapstoneState) -> dict:
        q_emb   = embedder.encode([state["question"]]).tolist()
        results = collection.query(query_embeddings=q_emb, n_results=3)

        chunks_list = results["documents"][0]
        metas       = results["metadatas"][0]
        topics      = [m.get("topic", "Physics") for m in metas]

        context = "\n\n---\n\n".join(
            f"[{topics[i]}]\n{chunks_list[i]}" for i in range(len(chunks_list))
        )
        return {"retrieved": context, "sources": topics}

    def skip_retrieval_node(state: CapstoneState) -> dict:
        return {"retrieved": "", "sources": []}

    # Node 4: Tool
    def tool_node(state: CapstoneState) -> dict:
        question = state["question"]
        intent   = state.get("intent", "")

        if intent == "search":
            tool_name = "web_search"
            try:
                from ddgs import DDGS
                with DDGS() as ddgs:
                    results = list(ddgs.text(question + " physics", max_results=3))
                if results:
                    tool_result = "\n".join(
                        f"{i+1}. {r.get('title','No title')}: {r.get('body','No summary')[:200]}"
                        for i, r in enumerate(results)
                    )
                else:
                    tool_result = "No relevant web results found."
            except Exception as e:
                tool_result = f"Web search error: {e}"

        elif intent == "numerical":
            tool_name   = "numerical_solver"
            tool_result = f"""
Solve the following physics numerical step-by-step:

Question: {question}

Include:
- Given data
- Formula used
- Substitution
- Final answer with unit
- Short explanation
"""
        elif intent == "plan":
            tool_name   = "study_plan_generator"
            tool_result = f"""
Create a structured physics study plan for:

{question}

Include:
- Important topics
- Recommended order of study
- Daily/weekly schedule
- Revision strategy
- Practice suggestions
"""
        elif intent == "derivation":
            tool_name   = "derivation_helper"
            tool_result = f"""
Explain the derivation of the following physics expression step-by-step:

{question}

Include:
- Starting formula
- Intermediate steps
- Final formula
- Meaning of each term
"""
        else:
            tool_name   = "concept_visualizer"
            tool_result = f"""
Explain the following physics concept clearly:

{question}

Include:
- Definition
- Key idea
- Real-life example
- Diagram description if needed
- Comparison with related concept if useful
"""

        return {"tool_name": tool_name, "tool_input": question, "tool_result": tool_result}

    # Node 5: Answer
    def answer_node(state: CapstoneState) -> dict:
        question     = state["question"]
        retrieved    = state.get("retrieved", "")
        tool_result  = state.get("tool_result", "")
        messages     = state.get("messages", [])
        eval_retries = state.get("eval_retries", 0)

        context_parts = []
        if retrieved:
            context_parts.append(f"KNOWLEDGE BASE:\n{retrieved}")
        if tool_result:
            context_parts.append(f"TOOL RESULT:\n{tool_result}")
        context = "\n\n".join(context_parts)

        if context:
            system_content = f"""
You are a Study Buddy for Physics.

Your job is to help students understand physics concepts, numericals, derivations, and study plans clearly and accurately.

STRICT RULES:
- Answer ONLY using the provided context (knowledge base or tool result)
- DO NOT use outside knowledge
- If the answer is not available in the context, say: "I don't have that information in my knowledge base."
- Keep the explanation clear, structured, and student-friendly
- Use steps, bullet points, and formulas when helpful

{context}
"""
        else:
            system_content = """
You are a Study Buddy for Physics.
Answer using only the conversation history. If unsure, say you don't know.
Be clear, simple, and student-friendly.
"""

        if eval_retries > 0:
            system_content += """

IMPORTANT: Your previous answer was not sufficiently grounded.
Now strictly ensure every statement comes from the provided context.
"""

        lc_msgs = [SystemMessage(content=system_content)]
        for msg in messages[:-1]:
            if msg["role"] == "user":
                lc_msgs.append(HumanMessage(content=msg["content"]))
            else:
                lc_msgs.append(AIMessage(content=msg["content"]))
        lc_msgs.append(HumanMessage(content=question))

        response = llm.invoke(lc_msgs)
        return {"answer": response.content}

    # Node 6: Eval
    def eval_node(state: CapstoneState) -> dict:
        answer  = state.get("answer", "")
        context = state.get("retrieved", "")[:500]
        retries = state.get("eval_retries", 0)

        if not context:
            return {"faithfulness": 1.0, "eval_retries": retries + 1}

        prompt = f"""Rate faithfulness: does this answer use ONLY information from the context?
Reply with ONLY a number between 0.0 and 1.0.

Context: {context}
Answer: {answer[:300]}"""

        result = llm.invoke(prompt).content.strip()
        try:
            score = float(result.split()[0].replace(",", "."))
            score = max(0.0, min(1.0, score))
        except Exception:
            score = 0.5

        return {"faithfulness": score, "eval_retries": retries + 1}

    # Node 7: Save
    def save_node(state: CapstoneState) -> dict:
        messages = state.get("messages", [])
        messages = messages + [{"role": "assistant", "content": state["answer"]}]
        return {"messages": messages}

    # ── Routing helpers ───────────────────────────────────
    def route_decision(state: CapstoneState) -> str:
        route = state.get("route", "retrieve")
        if route == "tool":
            return "tool"
        elif route == "memory_only":
            return "memory_only"
        return "retrieve"

    def eval_decision(state: CapstoneState) -> str:
        score   = state.get("faithfulness", 1.0)
        retries = state.get("eval_retries", 0)
        if score >= FAITHFULNESS_THRESHOLD or retries >= MAX_EVAL_RETRIES:
            return "save"
        return "answer"

    # ── Graph assembly ────────────────────────────────────
    builder = StateGraph(CapstoneState)

    builder.add_node("memory",      memory_node)
    builder.add_node("router",      router_node)
    builder.add_node("retrieve",    retrieval_node)
    builder.add_node("memory_only", skip_retrieval_node)
    builder.add_node("tool",        tool_node)
    builder.add_node("answer",      answer_node)
    builder.add_node("eval",        eval_node)
    builder.add_node("save",        save_node)

    builder.set_entry_point("memory")
    builder.add_edge("memory", "router")

    builder.add_conditional_edges(
        "router",
        route_decision,
        {"retrieve": "retrieve", "memory_only": "memory_only", "tool": "tool"},
    )

    builder.add_edge("retrieve",    "answer")
    builder.add_edge("memory_only", "answer")
    builder.add_edge("tool",        "answer")
    builder.add_edge("answer",      "eval")

    builder.add_conditional_edges(
        "eval",
        eval_decision,
        {"answer": "answer", "save": "save"},
    )

    builder.add_edge("save", END)

    checkpointer = MemorySaver()
    return builder.compile(checkpointer=checkpointer)
