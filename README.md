# ⚛️ Study Buddy — Physics
**Capstone Project | Sanchita Singh**

An AI-powered physics tutor built with LangGraph, ChromaDB, and Streamlit.

---

## 📁 Repository Structure

```
├── capstone_streamlit.py       ← Streamlit UI (main entry point)
├── agent.py                    ← Shared agent module (LangGraph graph)
├── requirements.txt            ← Python dependencies
├── Study_buddy_Sanchita_Singh_.ipynb  ← Original notebook
└── pdfs/                       ← Your physics PDF knowledge base
    ├── Damped Harmonic Motion.pdf
    ├── EM Waves Transverse Nature.pdf
    ├── Fraunhofer Diffraction.pdf
    ├── Inference of Light.pdf
    ├── Laser Components.pdf
    ├── Maxwells Equations.pdf
    ├── Quantum Process.pdf
    ├── Simple Harmonic Motion.pdf
    └── Waves and Wave Motion.pdf
```

---

## 🚀 Deploy on Streamlit Cloud (Step-by-Step)

### Step 1 — Push to GitHub
1. Create a new GitHub repository (public or private)
2. Upload ALL of the following:
   - `capstone_streamlit.py`
   - `agent.py`
   - `requirements.txt`
   - All your PDF files inside a folder called **`pdfs/`**
   - Your notebook file (optional but good for submission)

### Step 2 — Deploy on Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click **"New app"**
4. Select your repository
5. Set **Main file path** to: `capstone_streamlit.py`
6. Click **"Advanced settings"** → **Secrets** → Add:
   ```
   GROQ_API_KEY = "your_groq_api_key_here"
   ```
7. Click **Deploy!**

### Step 3 — Share your app
- Once deployed, you'll get a public URL like:
  `https://your-app-name.streamlit.app`
- Share this URL for submission ✅

---

## 🏃 Run Locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Create a .env file with your key
echo "GROQ_API_KEY=your_key_here" > .env

# 3. Run the app
streamlit run capstone_streamlit.py
```

---

## 🧠 Agent Capabilities

| Capability | Implementation |
|---|---|
| LangGraph StateGraph | 7-node graph (memory → router → retrieve/tool → answer → eval → save) |
| ChromaDB RAG | 9 physics PDFs, chunked & embedded with all-MiniLM-L6-v2 |
| Conversation memory | MemorySaver + thread_id, sliding window of 3 turns |
| Self-reflection | Eval node scores faithfulness; retries if score < 0.7 |
| Tool use | Web search, Numerical solver, Study plan generator, Concept visualizer, Derivation helper |
| Deployment | Streamlit UI |
