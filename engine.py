# --- FIX for ChromaDB/SQLite on Streamlit Cloud ---
import sys
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# --- END FIX ---

import os
import streamlit as st
from typing import Dict, Any, List, Tuple
import pandas as pd
import time

from pydantic import BaseModel, Field

# LangChain
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_cohere import CohereEmbeddings
from langchain_google_genai import GoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_cohere import ChatCohere
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

# PDF
from pypdf import PdfReader

# --- Model & Schema Definitions ---
MODEL_CONFIG = {
    "gemini": { "class": GoogleGenerativeAI, "args": { "model": "gemini-1.5-flash", "temperature": 0.1, "api_key_env": "GOOGLE_API_KEY" }},
    "deepseek": { "class": ChatOpenAI, "args": { "model": "deepseek/deepseek-chat", "temperature": 0.1, "base_url": "https://openrouter.ai/api/v1", "api_key_env": "OPENROUTER_API_KEY" }},
    "cohere": { "class": ChatCohere, "args": { "model": "command-r-plus", "temperature": 0.1, "api_key_env": "COHERE_API_KEY" }}
}

class ModelAnswer(BaseModel):
    answer: str
    confidence: int

# --- Core Processing Functions ---

def pdf_to_text_chunks(pdf_path: str, chunk_size=1500, chunk_overlap=100) -> List[str]:
    """Extract text from a PDF and split it into manageable chunks."""
    reader = PdfReader(pdf_path)
    text = "".join(page.extract_text() or "" for page in reader.pages)
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)

@st.cache_resource
def create_vectorstore(pdf_path: str, chunk_size=1500, chunk_overlap=100, k=5):
    """
    Build and cache the Chroma vectorstore retriever using API-based embeddings.
    NOW WITH BATCH PROCESSING to handle API rate limits.
    """
    chunks = pdf_to_text_chunks(pdf_path, chunk_size, chunk_overlap)

    cohere_api_key = os.getenv("COHERE_API_KEY")
    if not cohere_api_key:
        st.error("Cohere API key not found in secrets. It's required for embeddings.")
        return None

    embeddings = CohereEmbeddings(model="embed-english-v3.0", cohere_api_key=cohere_api_key)

    # --- BATCHING LOGIC START ---
    batch_size = 32 # Process 32 chunks at a time, a safe number for most APIs
    vectorstore = None

    # Create a progress bar for user feedback
    progress_bar = st.progress(0, text="Embedding document chunks...")

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]

        if vectorstore is None:
            # Create the vectorstore with the first batch
            vectorstore = Chroma.from_texts(texts=batch, embedding=embeddings)
        else:
            # Add subsequent batches to the existing vectorstore
            vectorstore.add_texts(texts=batch)

        # Update progress
        progress_percentage = min((i + batch_size) / len(chunks), 1.0)
        progress_bar.progress(progress_percentage, text=f"Embedding document chunks... {i+batch_size}/{len(chunks)}")

        # A small delay to respect rate limits even more
        time.sleep(0.5)

    progress_bar.empty() # Clear the progress bar on completion
    # --- BATCHING LOGIC END ---

    if vectorstore is None:
        st.error("Vectorstore could not be created. The document might be empty.")
        return None

    return vectorstore.as_retriever(search_kwargs={"k": k})

@st.cache_resource
def initialize_models_and_chains(_retriever, model_selection: List[str]) -> Tuple[Dict, Any]:
    """Initialize selected LLMs and their QA chains. Cached per retriever and model selection."""
    chains = {}

    for name in model_selection:
        if name in MODEL_CONFIG:
            config = MODEL_CONFIG[name]
            model_class = config["class"]
            args = config["args"].copy()

            api_key_env = args.pop("api_key_env")
            args["api_key"] = os.getenv(api_key_env)

            if not args["api_key"]:
                st.warning(f"API key for {name} not found. Skipping this model.")
                continue

            llm = model_class(**args)
            chain = RetrievalQA.from_chain_type(llm=llm, retriever=_retriever, chain_type="stuff")
            chains[name] = chain

    reconciler = ChatCohere(model="command-r-plus", temperature=0.1, cohere_api_key=os.getenv("COHERE_API_KEY"))
    return chains, reconciler

# --- Response Generation ---
def get_verified_response(question: str, retriever, chains: Dict[str, Any], reconciler: Any) -> Tuple[str, Dict[str, ModelAnswer]]:
    """Query all selected models and use a reconciler to generate a final verified answer."""
    per_model_answers = {}

    for model_name, chain in chains.items():
        try:
            result = chain.run(question)
            confidence = 75 if "not found" not in result.lower() else 25
            per_model_answers[model_name] = ModelAnswer(answer=result, confidence=confidence)
        except Exception as e:
            per_model_answers[model_name] = ModelAnswer(answer=f"Error processing with {model_name}: {e}", confidence=0)

    answers_context = "\n\n".join([f"--- Answer from {name} (Confidence: {res.confidence}) ---\n{res.answer}" for name, res in per_model_answers.items()])
    reconciler_prompt = f"""You are a senior financial analyst. Your task is to synthesize the best possible answer to a user's question based on responses from several AI models.
    User's Question: "{question}"
    Provided AI Answers:\n{answers_context}\n
    Your Task: Analyze all answers. Identify points of agreement and disagreement. Filter out errors. Construct a single, comprehensive, and well-structured final answer. Do not mention the other models. Present the final answer as your own expert analysis.
    Final Verified Answer:
    """

    try:
        reconciled_response = reconciler.invoke(reconciler_prompt)
        final_answer = reconciled_response.content
    except Exception as e:
        final_answer = f"**Reconciliation Failed:** {e}\n\n**Raw Answers:**\n{answers_context}"

    return final_answer, per_model_answers

# --- Market Data Utility ---
def get_market_snapshot_md(ticker_symbol: str) -> str:
    """Fetch and format a Markdown table of live market data for a given stock ticker."""
    import yfinance as yf
    try:
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info
        if not info or 'longName' not in info:
            return f"Could not retrieve valid market data for ticker '{ticker_symbol}'."

        def fmt(val):
            if val is None: return "N/A"
            try:
                n = float(val)
                if n >= 1e12: return f"{n / 1e12:.2f} T"
                if n >= 1e9:  return f"{n / 1e9:.2f} B"
                if n >= 1e6:  return f"{n / 1e6:.2f} M"
                return f"{int(n):,}"
            except (ValueError, TypeError): return "N/A"

        data = {
            "Metric": ["**Market Price**", "52-Week Range", "Market Cap", "P/E Ratio (TTM)", "Dividend Yield"],
            "Value": [
                f"**{info.get('regularMarketPrice', 'N/A')} {info.get('currency', '')}**",
                f"{info.get('fiftyTwoWeekLow', 'N/A')} - {info.get('fiftyTwoWeekHigh', 'N/A')}",
                f"{fmt(info.get('marketCap'))} {info.get('currency', '')}",
                f"{info.get('trailingPE', 0):.2f}" if isinstance(info.get('trailingPE'), float) else "N/A",
                f"{info.get('dividendYield', 0)*100:.2f}%" if isinstance(info.get('dividendYield'), float) else "N/A"
            ]
        }
        df = pd.DataFrame(data)
        return f"### Live Market Snapshot: {info.get('longName', 'N/A')} ({ticker_symbol.upper()})\n\n{df.to_markdown(index=False)}"
    except Exception as e:
        return f"An error occurred while fetching market data: {e}"
