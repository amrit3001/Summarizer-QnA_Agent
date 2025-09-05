# engine.py (Lite version for Streamlit Cloud)

import os
import hashlib
import pickle
from pathlib import Path
from typing import Dict, Any, List, Tuple

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

# --------------------------
# Default Model Config
# --------------------------
MODEL_CONFIG = {
    "gemini": {
        "class": GoogleGenerativeAI,
        "args": {
            "model": "gemini-2.5-flash",
            "temperature": 0.1,
            "api_key_env": "GOOGLE_API_KEY"
        }
    },
    "deepseek": {
        "class": ChatOpenAI,
        "args": {
            "model": "deepseek/deepseek-chat",
            "temperature": 0.1,
            "base_url": "https://openrouter.ai/api/v1",
            "api_key_env": "OPENROUTER_API_KEY"
        }
    },
    "cohere": {
        "class": ChatCohere,
        "args": {
            "model": "command-r-plus",
            "temperature": 0.1,
            "api_key_env": "COHERE_API_KEY"
        }
    }
}

# --------------------------
# Schema
# --------------------------
class ModelAnswer(BaseModel):
    answer: str
    confidence: int

# --------------------------
# Utilities
# --------------------------
def pdf_to_text_chunks(pdf_path: str, chunk_size=1500, chunk_overlap=100) -> List[str]:
    """Extract text from PDF and split into chunks."""
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_text(text)
    return chunks


def hash_file(file_path: str) -> str:
    """Return md5 hash of file (for caching)."""
    with open(file_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

# --------------------------
# Vectorstore
# --------------------------
def create_or_load_vectorstore(pdf_path: str, chunk_size=1500, chunk_overlap=100, k=5):
    """Build or load Chroma vectorstore from a PDF."""
    cache_dir = Path(".vector_cache")
    cache_dir.mkdir(exist_ok=True)
    pdf_hash = hash_file(pdf_path)
    cache_file = cache_dir / f"{pdf_hash}.pkl"

    if cache_file.exists():
        with open(cache_file, "rb") as f:
            retriever = pickle.load(f)
        return retriever

    chunks = pdf_to_text_chunks(pdf_path, chunk_size, chunk_overlap)
    embeddings = CohereEmbeddings(model="embed-english-v3.0", cohere_api_key=os.getenv("COHERE_API_KEY"))
    vectorstore = Chroma.from_texts(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    with open(cache_file, "wb") as f:
        pickle.dump(retriever, f)

    return retriever

# --------------------------
# Models & Chains
# --------------------------
def initialize_models_and_chains(model_subset, retriever, qa_prompt=None, reconciler_prompt=None):
    """Initialize selected LLMs and their QA chains."""
    chains = {}

    for name, config in model_subset.items():
        model_class = config["class"]
        args = config["args"].copy()

        if "api_key_env" in args:
            env_var = args.pop("api_key_env")
            args["api_key"] = os.getenv(env_var)

        llm = model_class(**args)
        chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff",
        )
        chains[name] = chain

    reconciler = ChatCohere(model="command-r-plus", temperature=0.1, cohere_api_key=os.getenv("COHERE_API_KEY"))
    return chains, reconciler

# --------------------------
# Verified Response
# --------------------------
def get_verified_response(question: str, retriever, chains: Dict[str, Any], reconciler: Any) -> Tuple[str, Dict[str, ModelAnswer]]:
    """Run the question across models and reconcile answers."""
    per_model_answers: Dict[str, ModelAnswer] = {}

    for model_name, chain in chains.items():
        try:
            result = chain.run(question)
            per_model_answers[model_name] = ModelAnswer(answer=result, confidence=70)
        except Exception as e:
            per_model_answers[model_name] = ModelAnswer(answer=f"Error: {e}", confidence=0)

    all_answers = "\n\n".join([f"{name}: {res.answer}" for name, res in per_model_answers.items()])
    reconciled_input = f"Question: {question}\nAnswers:\n{all_answers}\n\nProvide the best final answer as JSON with 'answer' and 'confidence'."

    try:
        reconciled = reconciler.invoke(reconciled_input)
        final_answer = reconciled.content
    except Exception as e:
        final_answer = f"Reconciliation failed: {e}"

    return final_answer, per_model_answers

# --------------------------
# Market Snapshot
# --------------------------
def market_snapshot_md(ticker_symbol: str) -> str:
    import yfinance as yf
    try:
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info
        if not info or 'longName' not in info:
            return f"Could not retrieve valid data for {ticker_symbol}."

        def fmt(n):
            try:
                n = float(n)
            except Exception:
                return "N/A"
            if n >= 1_000_000_000_000: return f"{n/1_000_000_000_000:.2f} T"
            if n >= 1_000_000_000: return f"{n/1_000_000_000:.2f} B"
            if n >= 1_000_000: return f"{n/1_000_000:.2f} M"
            return f"{int(n):,}"

        name = info.get('longName', 'N/A')
        currency = info.get('currency', '')
        cur = info.get('regularMarketPrice')
        prev = info.get('previousClose')
        hi = info.get('fiftyTwoWeekHigh')
        lo = info.get('fiftyTwoWeekLow')
        market_cap = fmt(info.get('marketCap'))
        vol = fmt(info.get('regularMarketVolume'))
        pe = info.get('trailingPE')
        dy = info.get('dividendYield')

        cur_s = "N/A" if cur is None else f"{cur} {currency}"
        prev_s = "N/A" if prev is None else f"{prev} {currency}"
        range_s = f"{lo} - {hi} {currency}" if (lo and hi) else "N/A"
        pe_s = f"{float(pe):.2f}" if isinstance(pe, (int, float)) else "N/A"
        dy_s = f"{float(dy)*100:.2f}%" if isinstance(dy, (int, float)) else "N/A"

        md = f"""
### Live Market Snapshot: {name} ({ticker_symbol.upper()})
| Metric | Value |
| :--- | :--- |
| **Current Market Price** | **{cur_s}** |
| Previous Close | {prev_s} |
| 52-Week Range | {range_s} |
| Market Capitalization | {market_cap} {currency} |
| Volume | {vol} |
| P/E Ratio (TTM) | {pe_s} |
| Dividend Yield | {dy_s} |
"""
        return md
    except Exception as e:
        return f"Market data fetch error: {e}"
