# execution.py (Lite version)

import streamlit as st
import tempfile
from engine import (
    create_or_load_vectorstore,
    initialize_models_and_chains,
    get_verified_response,
    market_snapshot_md,
    ModelAnswer,
    MODEL_CONFIG
)

# Load secrets into environment variables
import os
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

if "OPENROUTER_API_KEY" in st.secrets:
    os.environ["OPENROUTER_API_KEY"] = st.secrets["OPENROUTER_API_KEY"]

if "COHERE_API_KEY" in st.secrets:
    os.environ["COHERE_API_KEY"] = st.secrets["COHERE_API_KEY"]

st.set_page_config(page_title="Financial Report Q&A Assistant", layout="wide")
st.title("📊 Financial Report Q&A Assistant (Demo)")

# Sidebar
with st.sidebar:
    st.header("Upload & Settings")
    uploaded_file = st.file_uploader("Upload Annual Report (PDF)", type=["pdf"])
    ticker = st.text_input("Stock Ticker", value="RELIANCE.NS")
    st.markdown("---")
    st.subheader("Model toggles")
    use_gemini = st.checkbox("Gemini (Google)", value=True)
    use_deepseek = st.checkbox("DeepSeek (OpenRouter)", value=True)
    use_cohere = st.checkbox("Cohere", value=True)
    st.markdown("---")
    run_analysis_btn = st.button("📊 Generate Full Analysis")
    st.write("")
    st.subheader("Q&A")
    question_input = st.text_input("Ask a question from the uploaded report")
    ask_btn = st.button("💬 Ask")

# Session state
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "chains" not in st.session_state:
    st.session_state.chains = None
if "reconciler" not in st.session_state:
    st.session_state.reconciler = None
if "pdf_path" not in st.session_state:
    st.session_state.pdf_path = None

# Upload handling
if uploaded_file is not None and st.session_state.retriever is None:
    with st.spinner("Processing PDF & building vectorstore..."):
        t = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        t.write(uploaded_file.read())
        t.flush()
        t.close()
        st.session_state.pdf_path = t.name

        retriever = create_or_load_vectorstore(t.name, chunk_size=1500, chunk_overlap=100, k=5)
        st.session_state.retriever = retriever
        st.success("✅ Vectorstore ready (cached by PDF hash).")

# Model init
if st.session_state.retriever and st.session_state.chains is None:
    with st.spinner("Initializing LLMs and RAG chains..."):
        selected_models = {}
        if use_gemini: selected_models["gemini"] = MODEL_CONFIG["gemini"]
        if use_deepseek: selected_models["deepseek"] = MODEL_CONFIG["deepseek"]
        if use_cohere: selected_models["cohere"] = MODEL_CONFIG["cohere"]

        chains, reconciler = initialize_models_and_chains(selected_models, st.session_state.retriever, None, None)
        st.session_state.chains = chains
        st.session_state.reconciler = reconciler
        st.success("✅ Models & chains initialized.")

# Run analysis
if run_analysis_btn:
    if not st.session_state.retriever:
        st.warning("Upload a PDF first.")
    else:
        st.subheader("📈 Company Snapshot")
        st.markdown(market_snapshot_md(ticker))
        st.subheader("🔎 Automated Analysis")
        for title, instructions in {
            "Growth Analysis": "Find current and previous year 'Revenue' and 'EBITDA'/'EBIT' and compute YoY growth.",
            "Profitability Analysis": "Calculate margins and DuPont ROE.",
            "Liquidity and Solvency": "Current & Quick ratio, interest cover, net debt/EBITDA.",
            "Efficiency and Working Capital": "CCC, DIO, DSO, DPO.",
            "Cash Flow and Dividends": "FCF and payout ratio."
        }.items():
            with st.expander(title):
                final, per_model = get_verified_response(instructions, st.session_state.retriever, st.session_state.chains, st.session_state.reconciler)
                st.markdown(final)
                if st.checkbox(f"Show per-model answers for {title}", key=f"show_models_{title}"):
                    for m, ma in per_model.items():
                        st.markdown(f"**{m}** (confidence: {ma.confidence})\n\n{ma.answer}")

# Q&A
if ask_btn:
    if not st.session_state.retriever:
        st.warning("Upload a PDF first.")
    elif not question_input.strip():
        st.warning("Enter a question.")
    else:
        with st.spinner("Querying models..."):
            final, per_model = get_verified_response(question_input, st.session_state.retriever, st.session_state.chains, st.session_state.reconciler)
        st.subheader("✅ Verified Answer")
        st.markdown(final)
        if st.checkbox("Show per-model answers", key="show_per_model_q"):
            for m, ma in per_model.items():
                st.markdown(f"**{m}** (confidence: {ma.confidence})\n\n{ma.answer}")
