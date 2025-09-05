# --- FIX for ChromaDB/SQLite on Streamlit Cloud ---
import sys
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# --- END FIX ---

import streamlit as st
import os
import tempfile
from engine import (
    create_vectorstore,
    initialize_models_and_chains,
    get_verified_response,
    get_market_snapshot_md,
    MODEL_CONFIG
)

# --- Page Config & Load Secrets ---
st.set_page_config(page_title="Finanlyze AI", layout="wide", page_icon="📊")

# This block should be at the very top to ensure env variables are set
for key in ["GOOGLE_API_KEY", "OPENROUTER_API_KEY", "COHERE_API_KEY"]:
    if key in st.secrets:
        os.environ[key] = st.secrets[key]

# --- Session State Initialization ---
if "app_state" not in st.session_state:
    st.session_state.app_state = {
        "retriever": None,
        "chains": None,
        "reconciler": None,
        "pdf_path": None,
        "ticker": "MSFT" # Default ticker
    }

# --- UI: Sidebar for Setup & Controls ---
with st.sidebar:
    st.header("Setup")
    st.markdown("Upload a financial report PDF and select the AI models to power the analysis.")

    uploaded_file = st.file_uploader("Upload Annual Report", type=["pdf"])
    ticker_input = st.text_input("Enter Stock Ticker", value=st.session_state.app_state["ticker"])

    st.markdown("---")
    st.subheader("Model Selection")
    model_selection = [name for name in MODEL_CONFIG if st.checkbox(name.capitalize(), value=True)]

    if st.button("Process Document & Initialize Models", use_container_width=True):
        if uploaded_file and ticker_input and model_selection:
            with st.spinner("Processing PDF and warming up AI models..."):
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tfile:
                    tfile.write(uploaded_file.read())
                    st.session_state.app_state["pdf_path"] = tfile.name

                # Update ticker
                st.session_state.app_state["ticker"] = ticker_input

                # Core processing steps
                st.session_state.app_state["retriever"] = create_vectorstore(st.session_state.app_state["pdf_path"])

                if st.session_state.app_state["retriever"]:
                    chains, reconciler = initialize_models_and_chains(st.session_state.app_state["retriever"], model_selection)
                    st.session_state.app_state["chains"] = chains
                    st.session_state.app_state["reconciler"] = reconciler
                    st.success("Ready to analyze!")
                else:
                    st.error("Failed to process the document.")
        else:
            st.warning("Please upload a file, enter a ticker, and select at least one model.")

# --- Main Content Area ---
st.title("📊 Finanlyze AI: Financial Report Assistant")

if not st.session_state.app_state.get("chains"):
    st.info("Welcome! Please upload a document and initialize the models using the sidebar to begin.")
else:
    st.header("Company Snapshot")
    st.markdown(get_market_snapshot_md(st.session_state.app_state["ticker"]))
    st.markdown("---")

    tab1, tab2 = st.tabs(["Automated Analysis", "Interactive Q&A"])

    with tab1:
        st.subheader("Generate a Report")
        analysis_prompts = {
            "Growth Analysis": "Find current and previous year 'Revenue' and 'EBITDA'/'EBIT', then compute YoY growth. Quote numbers and show calculations.",
            "Profitability Analysis": "Calculate Gross, Operating, and Net Profit margins for the most recent year. Also, perform a DuPont ROE analysis.",
            "Liquidity & Solvency": "Calculate the Current Ratio, Quick Ratio, Interest Coverage Ratio, and Net Debt to EBITDA ratio for the most recent year.",
        }
        for title, prompt in analysis_prompts.items():
            if st.button(f"Analyze {title}", key=title, use_container_width=True):
                with st.expander(title, expanded=True):
                    with st.spinner(f"Running {title}..."):
                        final_answer, per_model = get_verified_response(
                            prompt,
                            st.session_state.app_state["retriever"],
                            st.session_state.app_state["chains"],
                            st.session_state.app_state["reconciler"]
                        )
                        st.markdown(final_answer)
                        if st.checkbox("Show individual model answers", key=f"details_{title}"):
                            st.json({k: v.dict() for k, v in per_model.items()})
    with tab2:
        st.subheader("Ask a Custom Question")
        question_input = st.text_input("Enter your question about the report:", key="qa_input")
        if st.button("Get Answer", key="qa_button"):
            if question_input.strip():
                with st.spinner("Querying models and verifying answer..."):
                    final_answer, per_model = get_verified_response(
                        question_input,
                        st.session_state.app_state["retriever"],
                        st.session_state.app_state["chains"],
                        st.session_state.app_state["reconciler"]
                    )
                    st.markdown("#### Verified Answer")
                    st.info(final_answer)
                    with st.expander("View individual model answers"):
                        st.json({k: v.dict() for k, v in per_model.items()})
            else:
                st.warning("Please enter a question.")
