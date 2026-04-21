"""
capstone_streamlit.py — Study Buddy (Physics) Streamlit UI
Sanchita Singh | Capstone Project

Run locally:
    streamlit run capstone_streamlit.py

Deploy on Streamlit Cloud:
    1. Push this file + agent.py + requirements.txt + your PDF files to GitHub
    2. Go to share.streamlit.io → New app → point to this file
    3. Add GROQ_API_KEY in the Secrets section (Settings → Secrets)
"""

import os
import uuid
import streamlit as st

# ── PDF paths — update these to match your repo structure ──
PDF_FILES = [
    "pdfs/Damped Harmonic Motion.pdf",
    "pdfs/EM Waves Transverse Nature.pdf",
    "pdfs/Fraunhofer Diffraction.pdf",
    "pdfs/Inference of Light.pdf",
    "pdfs/Laser Components.pdf",
    "pdfs/Maxwells Equations.pdf",
    "pdfs/Quantum Process.pdf",
    "pdfs/Simple Harmonic Motion.pdf",
    "pdfs/Waves and Wave Motion.pdf",
]

KB_TOPICS = [
    "Damped Harmonic Motion",
    "EM Waves",
    "Fraunhofer Diffraction",
    "Interference of Light",
    "Laser Components",
    "Maxwell's Equations",
    "Quantum Processes",
    "Simple Harmonic Motion",
    "Waves & Wave Motion",
]

# ── Page config ────────────────────────────────────────────
st.set_page_config(
    page_title="Study Buddy — Physics",
    page_icon="⚛️",
    layout="centered",
)

st.title("⚛️ Study Buddy — Physics")
st.caption("Your AI-powered physics tutor | Built by Sanchita Singh")

# ── Load GROQ key from Streamlit secrets or environment ────
groq_key = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", ""))
if not groq_key:
    st.error("❌ GROQ_API_KEY not found. Add it to Streamlit Secrets or your .env file.")
    st.stop()
os.environ["GROQ_API_KEY"] = groq_key


# ── Load agent (cached so it runs only once per session) ───
@st.cache_resource(show_spinner="Loading knowledge base and agent…")
def load_agent():
    from agent import build_knowledge_base, build_agent
    embedder, collection = build_knowledge_base(PDF_FILES)
    agent_app = build_agent(embedder, collection)
    return agent_app, collection


try:
    agent_app, collection = load_agent()
    st.success(f"✅ Knowledge base loaded — {collection.count()} chunks")
except Exception as e:
    st.error(f"Failed to load agent: {e}")
    st.stop()


# ── Session state ──────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())[:8]


# ── Sidebar ────────────────────────────────────────────────
with st.sidebar:
    st.header("📚 About")
    st.write(
        "Study Buddy helps students understand physics concepts, "
        "solve numericals, explain derivations, and create study plans — "
        "all based on your curated knowledge base."
    )
    st.write(f"**Session ID:** `{st.session_state.thread_id}`")
    st.divider()

    st.write("**Topics covered:**")
    for topic in KB_TOPICS:
        st.write(f"• {topic}")

    st.divider()
    st.write("**Try asking:**")
    st.write("- What is simple harmonic motion?")
    st.write("- Solve: A spring has k=200 N/m and mass=2 kg. Find time period.")
    st.write("- Give me a study plan for waves.")
    st.write("- Derive the equation for damped oscillations.")

    if st.button("🗑️ New conversation"):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())[:8]
        st.rerun()


# ── Display chat history ───────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])


# ── Chat input ─────────────────────────────────────────────
if prompt := st.chat_input("Ask a physics question…"):
    with st.chat_message("user"):
        st.write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            result = agent_app.invoke({"question": prompt}, config=config)
            answer = result.get("answer", "Sorry, I could not generate an answer.")

        st.write(answer)

        faith   = result.get("faithfulness", 0.0)
        sources = result.get("sources", [])
        tool    = result.get("tool_name", "")

        meta_parts = []
        if faith > 0:
            meta_parts.append(f"Faithfulness: {faith:.2f}")
        if sources:
            meta_parts.append(f"Sources: {', '.join(sources)}")
        if tool and tool != "none":
            meta_parts.append(f"Tool: {tool}")
        if meta_parts:
            st.caption(" | ".join(meta_parts))

    st.session_state.messages.append({"role": "assistant", "content": answer})
