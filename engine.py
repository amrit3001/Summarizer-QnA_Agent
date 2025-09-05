# utils.py
import os
import re
import pickle
import hashlib
import tempfile
from typing import Dict, Any, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# LangChain + other imports
from pydantic import BaseModel, Field
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import PydanticOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.retrievers import BM25Retriever, EnsembleRetriever

# LLMs
from langchain_google_genai import GoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_cohere import ChatCohere

# ---------- Default Model Config ----------
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



# ---------- Config (modify if needed) ----------
EMBEDDING_MODEL_NAME_DEFAULT = "sentence-transformers/all-MiniLM-L6-v2"  # recommended for speed
VECTORSTORE_DIR = Path("vectorstores")
VECTORSTORE_DIR.mkdir(exist_ok=True)

# ---------- Pydantic model ----------
class ModelAnswer(BaseModel):
    answer: str
    confidence: int

pydantic_parser = PydanticOutputParser(pydantic_object=ModelAnswer)

# ---------- Reconciler Schema ----------
class ReconcilerOutput(BaseModel):
    answer: str = Field(..., description="The final reconciled answer")
    confidence: int = Field(..., description="Confidence score (0-100)")


# ---------- Utility helpers ----------
def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def save_bm25(obj, path: Path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def load_bm25(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)

# ---------- Preprocess ----------
def preprocess_documents(docs: List[Document]) -> List[Document]:
    if not docs:
        return docs
    text = docs[0].page_content
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
    text = re.sub(r'(?i)\bpage\b\s*\d+\s*(of\s*\d+)?', '', text)
    text = re.sub(r'(?i)\bannual report\b\s*\d{4}', '', text)
    text = re.sub(r'[₹$,]', '', text)
    text = re.sub(r'\s+', ' ', text)
    docs[0].page_content = text.strip()
    return docs

# ---------- Create or load vectorstore with disk cache ----------
def create_or_load_vectorstore(
    pdf_path: str,
    embedding_model: str = EMBEDDING_MODEL_NAME_DEFAULT,
    chunk_size: int = 1500,
    chunk_overlap: int = 100,
    k: int = 5
) -> EnsembleRetriever:
    # compute pdf hash
    with open(pdf_path, "rb") as f:
        b = f.read()
    pdf_hash = sha256_bytes(b)
    folder = VECTORSTORE_DIR / pdf_hash
    faiss_dir = folder / "faiss"
    bm25_path = folder / "bm25.pkl"

    # if cached, load
    if folder.exists() and faiss_dir.exists() and bm25_path.exists():
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        faiss_vs = FAISS.load_local(str(faiss_dir), embeddings, allow_dangerous_deserialization=True)
        faiss_retriever = faiss_vs.as_retriever(search_kwargs={"k": k})
        bm25 = load_bm25(bm25_path)
        bm25.k = k
        return EnsembleRetriever(retrievers=[faiss_retriever, bm25], weights=[0.6, 0.4])

    # else, create
    loader = UnstructuredPDFLoader(pdf_path, mode="single", strategy="fast")
    documents = loader.load()
    documents = preprocess_documents(documents)
    full_text = documents[0].page_content

    section_pattern = r"(?i)(^\s*(?:consolidated|standalone)?\s*(?:balance sheet|statement of profit and loss|cash flow statement|notes to the financial statements|management discussion and analysis|independent auditor.s report)\s*$)"
    split_parts = re.split(section_pattern, full_text, flags=re.MULTILINE)

    semantic_chunks = []
    current_title = "General Report"
    for i in range(1, len(split_parts), 2):
        content = split_parts[i-1]
        title = split_parts[i].strip()
        if content.strip():
            semantic_chunks.append(Document(page_content=content, metadata={"source": pdf_path, "section": current_title}))
        current_title = title
    final_content = split_parts[-1]
    if final_content.strip():
        semantic_chunks.append(Document(page_content=final_content, metadata={"source": pdf_path, "section": current_title}))

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name="gpt-4", chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    final_chunks = text_splitter.split_documents(semantic_chunks)

    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    faiss_vs = FAISS.from_documents(final_chunks, embeddings)
    faiss_retriever = faiss_vs.as_retriever(search_kwargs={"k": k})

    bm25 = BM25Retriever.from_documents(final_chunks)
    bm25.k = k

    # save to disk
    folder.mkdir(parents=True, exist_ok=True)
    faiss_vs.save_local(str(folder / "faiss"))
    save_bm25(bm25, bm25_path)

    return EnsembleRetriever(retrievers=[faiss_retriever, bm25], weights=[0.6, 0.4])

# ---------- RAG chain factory ----------
def create_rag_chain(llm, retriever, prompt: PromptTemplate):
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | pydantic_parser
    )

# ---------- Initialize models & chains ----------
def initialize_models_and_chains(model_config: Dict, retriever, qa_prompt: PromptTemplate, reconciler_prompt: PromptTemplate) -> Tuple[Dict, Any]:
    chains = {}
    first_llm_for_reconciler = None
    for name, config in model_config.items():
        try:
            args = config['args'].copy()
            api_key_env = args.pop('api_key_env', None)
            if api_key_env:
                args['api_key'] = os.environ.get(api_key_env)
            llm = config['class'](**args)
            if first_llm_for_reconciler is None:
                first_llm_for_reconciler = llm
            chains[name] = create_rag_chain(llm, retriever, qa_prompt)
        except Exception as e:
            chains[name] = None
    if first_llm_for_reconciler:
        reconciliation_chain = reconciler_prompt | first_llm_for_reconciler | pydantic_parser
    else:
        reconciliation_chain = None
    return chains, reconciliation_chain

# ---------- Parallel query + reconciliation ----------
def get_verified_response(
    question: str,
    retriever,
    chains: Dict[str, Any],
    reconciler_chain: Any,
    max_workers: int = 3
) -> Tuple[str, Dict[str, ModelAnswer]]:
    # retrieve context once
    context_docs = retriever.get_relevant_documents(question)
    if not context_docs:
        return "Could not retrieve any relevant context from the document for this question.", {}
    formatted_context = "\n\n".join(doc.page_content for doc in context_docs)

    results: Dict[str, ModelAnswer] = {}
    # run chains in parallel (invoke each chain with the question)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {}
        for name, chain in chains.items():
            if chain is None:
                # chain failed earlier; mark as error
                results[name] = ModelAnswer(answer="Model chain not initialized", confidence=0)
                continue
            futures[ex.submit(chain.invoke, question)] = name

        for fut in as_completed(futures):
            name = futures[fut]
            try:
                ma = fut.result()
                # Ensure it's ModelAnswer (pydantic) or convert
                if isinstance(ma, ModelAnswer):
                    results[name] = ma
                else:
                    # chain returned a pydantic object compatible with ModelAnswer
                    try:
                        results[name] = ModelAnswer(**ma.dict())  # if it's a pydantic model
                    except Exception:
                        results[name] = ModelAnswer(answer=str(ma), confidence=0)
            except Exception as e:
                results[name] = ModelAnswer(answer=f"Model error: {e}", confidence=0)

    # format for reconciler
    all_answers_formatted = "\n\n".join(
        f"--- Answer from Model: {name} (Confidence: {ma.confidence}) ---\n{ma.answer}"
        for name, ma in results.items()
    )

    if reconciler_chain is None:
        # just return concatenated model outputs and best-guess
        final = "No reconciler available. Model outputs:\n\n" + all_answers_formatted
    else:
        final = reconciler_chain.invoke({
            "question": question,
            "context": formatted_context,
            "all_answers": all_answers_formatted
        })

    return final, results

# ---------- Market snapshot generator (returns markdown string) ----------
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
