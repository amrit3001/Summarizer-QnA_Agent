import streamlit as st
import os
import tempfile
from engine import (
    load_local_api_keys,
    create_vectorstore_local,
    initialize_models_and_chains,
    get_verified_response,
    get_market_snapshot_md,
    MODEL_CONFIG
)

# --- Load API keys from .env file at the very start ---
load_local_api_keys()

# --- Page Config & State ---
st.set_page_config(page_title="Finanlyze AI (Local)", layout="wide", page_icon="📊")

if "app_state" not in st.session_state:
    st.session_state.app_state = {
        "retriever": None,
        "chains": None,
        "reconciler": None,
        "ticker": "MSFT" # Default ticker
    }

# --- UI Sidebar ---
with st.sidebar:
    st.header("Setup (Local Mode)")
    st.markdown("This app runs locally, using your computer's resources for embedding.")
    
    uploaded_file = st.file_uploader("Upload Annual Report", type=["pdf"])
    ticker_input = st.text_input("Enter Stock Ticker", value=st.session_state.app_state["ticker"])
    
    st.markdown("---")
    st.subheader("Model Selection")
    model_selection = [name for name in MODEL_CONFIG if st.checkbox(name.capitalize(), value=True)]

    if st.button("Process Document", use_container_width=True):
        if uploaded_file and ticker_input and model_selection:
            with st.spinner("Processing PDF on your local machine... This might take a moment the first time as the model downloads."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tfile:
                    tfile.write(uploaded_file.read())
                    pdf_path = tfile.name
                
                st.session_state.app_state["ticker"] = ticker_input
                st.session_state.app_state["retriever"] = create_vectorstore_local(pdf_path)
                
                if st.session_state.app_state["retriever"]:
                    chains, reconciler = initialize_models_and_chains(st.session_state.app_state["retriever"], model_selection)
                    st.session_state.app_state["chains"] = chains
                    st.session_state.app_state["reconciler"] = reconciler
                    st.success("Ready to analyze!")
        else:
            st.warning("Please provide a file, ticker, and select at least one model.")

# --- Main Content ---
st.title("📊 Finanlyze AI: Local Financial Assistant")

if not st.session_state.app_state.get("chains"):
    st.info("Welcome! Upload a document and process it to begin.")
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
