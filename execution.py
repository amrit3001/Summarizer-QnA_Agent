# execution.py

import streamlit as st
import tempfile
from pathlib import Path
import os

from engine import (
    MODEL_CONFIG,
    create_or_load_vectorstore,
    initialize_models_and_chains,
    get_verified_response,
    market_snapshot_md,
    ReconcilerOutput
)

from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser


# --------------------------
# Defaults
# --------------------------
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
chunk_size = 1500
chunk_overlap = 100
k = 5


# --------------------------
# Prompt definitions
# --------------------------
qa_prompt = PromptTemplate.from_template(
    """
You are a financial assistant. Use the provided context to answer the question.

Context:
{context}

Question: {question}

Answer:
"""
)

parser = PydanticOutputParser(pydantic_object=ReconcilerOutput)

reconciler_prompt = PromptTemplate(
    template="""
You are reconciling multiple financial model answers.

Question: {question}
Context: {context}
All Answers: {all_answers}

Provide the best consolidated answer as JSON.
{format_instructions}
""",
    input_variables=["question", "context", "all_answers"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)


# --------------------------
# Streamlit UI
# --------------------------
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


# --------------------------
# Session state
# --------------------------
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "chains" not in st.session_state:
    st.session_state.chains = None
if "reconciler" not in st.session_state:
    st.session_state.reconciler = None
if "pdf_path" not in st.session_state:
    st.session_state.pdf_path = None


# --------------------------
# Handle PDF upload
# --------------------------
if uploaded_file is not None and st.session_state.retriever is None:
    with st.spinner("Processing PDF & building vectorstore (this may take a moment)..."):
        t = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        t.write(uploaded_file.read())
        t.flush()
        t.close()
        st.session_state.pdf_path = t.name

        retriever = create_or_load_vectorstore(
            t.name,
            embedding_model=embedding_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            k=k,
        )
        st.session_state.retriever = retriever
        st.success("✅ Vectorstore ready (cached by PDF hash).")


# --------------------------
# Initialize models & chains
# --------------------------
if st.session_state.retriever and st.session_state.chains is None:
    with st.spinner("Initializing LLMs and RAG chains..."):
        # build config dynamically based on toggles
        model_subset = {}
        if use_gemini:
            model_subset["gemini"] = MODEL_CONFIG["gemini"]
        if use_deepseek:
            model_subset["deepseek"] = MODEL_CONFIG["deepseek"]
        if use_cohere:
            model_subset["cohere"] = MODEL_CONFIG["cohere"]

        chains, reconciler = initialize_models_and_chains(
            model_subset, st.session_state.retriever, qa_prompt, reconciler_prompt
        )
        st.session_state.chains = chains
        st.session_state.reconciler = reconciler
        st.success("✅ Models & chains initialized.")


# --------------------------
# Full analysis
# --------------------------
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
                final, per_model = get_verified_response(
                    instructions, st.session_state.retriever, st.session_state.chains, st.session_state.reconciler
                )
                try:
                    parsed_result = parser.parse(final)
                    st.markdown(parsed_result.answer)
                    st.caption(f"Confidence: {parsed_result.confidence}")
                except Exception as e:
                    st.markdown(final)
                    st.error(f"Parser failed: {e}")

                if st.checkbox(f"Show per-model answers for {title}", key=f"show_models_{title}"):
                    for m, ma in per_model.items():
                        st.markdown(f"**{m}** (confidence: {ma.confidence})\n\n{ma.answer}")


# --------------------------
# Q&A Section
# --------------------------
if ask_btn:
    if not st.session_state.retriever:
        st.warning("Upload a PDF first.")
    elif not question_input.strip():
        st.warning("Enter a question.")
    else:
        with st.spinner("Querying models..."):
            final, per_model = get_verified_response(
                question_input, st.session_state.retriever, st.session_state.chains, st.session_state.reconciler
            )
        try:
            parsed_result = parser.parse(final)
            final_answer = parsed_result.answer
            confidence = parsed_result.confidence
        except Exception as e:
            final_answer = final
            confidence = "N/A"
            st.error(f"Parser failed, showing raw output. Error: {e}")

        st.subheader("✅ Verified Answer")
        st.markdown(final_answer)
        st.caption(f"Confidence: {confidence}")

        if st.checkbox("Show per-model answers", key="show_per_model_q"):
            for m, ma in per_model.items():
                st.markdown(f"**{m}** (confidence: {ma.confidence})\n\n{ma.answer}")
