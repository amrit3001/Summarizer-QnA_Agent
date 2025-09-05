import os
from typing import Dict, Any, List, Tuple
import pandas as pd
import time
from dotenv import load_dotenv

import streamlit as st
from pydantic import BaseModel, Field

# LangChain
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_google_genai import GoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_cohere import ChatCohere
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# PDF
from pypdf import PdfReader

# --- Model & Schema Definitions ---
MODEL_CONFIG = {
    "gemini": { "class": GoogleGenerativeAI, "args": { "model": "gemini-1.5-flash", "temperature": 0.1, "api_key_env": "GOOGLE_API_KEY" }},
    "deepseek": { "class": ChatOpenAI, "args": { "model": "deepseek/deepseek-chat", "temperature": 0.1, "base_url": "https://openrouter.ai/api/v1", "api_key_env": "OPENROUTER_API_KEY" }},
    "cohere": { "class": ChatCohere, "args": { "model": "command-r-plus", "temperature": 0.1, "api_key_env": "COHERE_API_KEY" }}
}
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

class ModelAnswer(BaseModel):
    answer: str
    confidence: int

# --- API Key Loading ---
def load_local_api_keys():
    """Loads API keys from a local .env file."""
    load_dotenv()

# --- Core Processing Functions ---
def pdf_to_text_chunks(pdf_path: str, chunk_size=1500, chunk_overlap=100) -> List[str]:
    """Extract text from a PDF and split it into manageable chunks."""
    reader = PdfReader(pdf_path)
    text = "".join(page.extract_text() or "" for page in reader.pages)
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)

@st.cache_resource
def create_vectorstore_local(pdf_path: str, chunk_size=1500, chunk_overlap=100, k=5):
    """
    Build and cache the FAISS vectorstore retriever using a local sentence-transformer model.
    This runs on your CPU/GPU and makes no API calls.
    """
    chunks = pdf_to_text_chunks(pdf_path, chunk_size, chunk_overlap)
    
    # Use a local embedding model
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    
    # Use FAISS, which is highly efficient for local, in-memory vectorstores.
    vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
    
    return vectorstore.as_retriever(search_kwargs={"k": k})

@st.cache_resource
def initialize_models_and_chains(_retriever, model_selection: List[str]) -> Tuple[Dict, Any]:
    """Initialize selected LLMs and their QA chains."""
    chains = {}
    for name in model_selection:
        if name in MODEL_CONFIG:
            config = MODEL_CONFIG[name]
            args = config["args"].copy()
            api_key_env = args.pop("api_key_env")
            args["api_key"] = os.getenv(api_key_env)
            if not args["api_key"]:
                st.warning(f"API key for {name} not found in .env file. Skipping.")
                continue
            llm = config["class"](**args)
            chain = RetrievalQA.from_chain_type(llm=llm, retriever=_retriever, chain_type="stuff")
            chains[name] = chain

    reconciler = ChatCohere(model="command-r-plus", temperature=0.1, cohere_api_key=os.getenv("COHERE_API_KEY"))
    return chains, reconciler

# --- Response Generation (Sequential to avoid Q&A rate limits) ---
def get_verified_response(question: str, retriever, chains: Dict[str, Any], reconciler: Any) -> Tuple[str, Dict[str, ModelAnswer]]:
    """Query all selected models SEQUENTIALLY to respect API limits."""
    per_model_answers = {}
    models_to_query = list(chains.items())
    
    for i, (model_name, chain) in enumerate(models_to_query):
        try:
            result = chain.run(question)
            confidence = 75 if "not found" not in result.lower() else 25
            per_model_answers[model_name] = ModelAnswer(answer=result, confidence=confidence)
        except Exception as e:
            per_model_answers[model_name] = ModelAnswer(answer=f"Error: {e}", confidence=0)
        time.sleep(1)

    answers_context = "\n\n".join([f"--- Answer from {name} (Confidence: {res.confidence}) ---\n{res.answer}" for name, res in per_model_answers.items()])
    reconciler_prompt = f"""Synthesize the best answer from the following AI responses.
    User Question: "{question}"
    AI Answers:\n{answers_context}\n
    Final Verified Answer:
    """
    try:
        reconciled_response = reconciler.invoke(reconciler_prompt)
        final_answer = reconciled_response.content
    except Exception as e:
        final_answer = f"**Reconciliation Failed:** {e}"
    return final_answer, per_model_answers

# --- Market Data Utility ---
def get_market_snapshot_md(ticker_symbol: str) -> str:
    """Fetch and format a Markdown table of live market data."""
    import yfinance as yf
    try:
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info
        if not info or 'longName' not in info: return f"Could not retrieve data for '{ticker_symbol}'."
        def fmt(val):
            if val is None: return "N/A"
            try:
                n = float(val)
                if n >= 1e12: return f"{n / 1e12:.2f} T"
                if n >= 1e9:  return f"{n / 1e9:.2f} B"
                if n >= 1e6:  return f"{n / 1e6:.2f} M"
                return f"{int(n):,}"
            except (ValueError, TypeError): return "N/A"
        data = { "Metric": ["**Market Price**", "52-Week Range", "Mkt Cap"], "Value": [ f"**{info.get('regularMarketPrice', 'N/A')} {info.get('currency', '')}**", f"{info.get('fiftyTwoWeekLow', 'N/A')} - {info.get('fiftyTwoWeekHigh', 'N/A')}", f"{fmt(info.get('marketCap'))} {info.get('currency', '')}" ] }
        df = pd.DataFrame(data)
        return f"### {info.get('longName', 'N/A')} ({ticker_symbol.upper()})\n\n{df.to_markdown(index=False)}"
    except Exception as e: return f"Error fetching market data: {e}"
