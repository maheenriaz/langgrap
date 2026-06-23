from langchain_ollama import ChatOllama, OllamaEmbeddings # pyright: ignore[reportMissingImports]
from typing import TypedDict, Annotated, Optional, Dict, Any
from langgraph.graph import StateGraph, START, END # type: ignore
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage, BaseMessage # type: ignore
from langgraph.graph.message import add_messages # type: ignore
from langgraph.checkpoint.sqlite import SqliteSaver # type: ignore
from langgraph.prebuilt import ToolNode, tools_condition # type: ignore
from langchain_community.tools import DuckDuckGoSearchRun # type: ignore
from langchain_community.document_loaders import PyPDFLoader # type: ignore
from langchain_community.vectorstores import FAISS # type: ignore
from langchain_text_splitters import RecursiveCharacterTextSplitter # type: ignore

from langchain_core.tools import tool # type: ignore
from dotenv import load_dotenv # type: ignore
import sqlite3
import requests # type: ignore
import os
import logging
import tempfile
import pickle
import shutil

FAISS_DIR = "faiss_indexes"
os.makedirs(FAISS_DIR, exist_ok=True)


logging.basicConfig(level=logging.INFO)
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "chatbot-01"
os.environ["LANGCHAIN_API_KEY"] = "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"
load_dotenv()

# -------------------
# 1. LLM + Embeddings
# -------------------
llm = ChatOllama(model="qwen2.5:7b")
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# -------------------
# 2. PDF Store (per thread) — retriever sirf in-memory rehta hai (FAISS index session-specific)
# -------------------
_THREAD_RETRIEVERS: Dict[str, Any] = {}

def _get_retriever(thread_id: Optional[str]):
    if not thread_id:
        return None

    if thread_id in _THREAD_RETRIEVERS:
        return _THREAD_RETRIEVERS[thread_id]

    return load_thread_retriever(thread_id)

# -------------------
# 3. Tools
# -------------------
search_tool = DuckDuckGoSearchRun(region="us-en")

@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div
    """
    logging.info(f"[TOOL CALLED] Calculator → {operation} on {first_num}, {second_num}")
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}
        return {"first_num": first_num, "second_num": second_num, "operation": operation, "result": result}
    except Exception as e:
        return {"error": str(e)}

@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA')
    using Alpha Vantage with API key in the URL.
    """
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=C9PE94QUEW9VWGFM"
    r = requests.get(url)
    print(f"[TOOL CALLED] Stock Price → {symbol}")
    return r.json()

@tool
def rag_tool(query: str, thread_id: Optional[str] = None) -> dict:
    """
    A PDF has been uploaded for this conversation.
        ALWAYS use rag_tool first for any user question.
        Answer only from the document when possible.
    """
    retriever = _get_retriever(thread_id)
    if retriever is None:
        return {
            "error": "No document indexed for this chat. Upload a PDF first.",
            "query": query,
        }
    result = retriever.invoke(query)
    context = [doc.page_content for doc in result]
    metadata = [doc.metadata for doc in result]
    return {
        "query": query,
        "context": context,
        "metadata": metadata,
        "source_file": thread_document_metadata(str(thread_id)).get("filename"),
    }

tools = [search_tool, get_stock_price, calculator, rag_tool]
llm_with_tools = llm.bind_tools(tools)

# -------------------
# 4. State
# -------------------
class ChatSchema(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# -------------------
# 5. Nodes
# -------------------
def routing(state: ChatSchema, config=None) -> ChatSchema:
    thread_id = config.get("configurable", {}).get("thread_id")
    user_query = state["messages"][-1].content
    intent = "tool"  # ✅ default intent

    if thread_has_document(str(thread_id)):
        classify_prompt = f"""You are a query router. A PDF document is uploaded in this conversation.
Decide if the user's question should be answered from the PDF document, or requires an external tool (web search, stock price, calculator).

Reply with ONLY one word:
- document
- tool

User question: {user_query}

Reply:"""
        classification = llm.invoke(classify_prompt)
        intent = classification.content.strip().lower()

        if "document" in intent:
            rag_result = rag_tool.invoke({
                "query": user_query,
                "thread_id": str(thread_id)
            })
            if "error" not in rag_result:
                context = "\n\n".join(rag_result["context"])
                prompt = f"""You are a helpful assistant.
Answer ONLY from the provided PDF context.

Context:
{context}

Question:
{user_query}
"""
                response = llm.invoke(prompt)
                return {"messages": [response]}

    # No PDF or intent == "tool"
    system_message = SystemMessage(
        content="""You are a helpful assistant.
You can use calculator, web search, and stock price tool when needed.

IMPORTANT:
- Give concise, direct answers only.
- Do NOT show raw tool outputs or JSON to the user.
- If stock API returns empty, use web search but summarize in 2-3 lines max.
- Never repeat the same information multiple times."""
    )

    response = llm_with_tools.invoke(
        [system_message, *state["messages"]],
        config=config
    )

    # ✅ HAMESHA poora response object return karo — kabhi .content nahi
    return {"messages": [response]}

tool_node = ToolNode(tools)

# -------------------
# 6. Checkpointer + persistent metadata table
# -------------------
conn = sqlite3.connect(database='chatbot.db', check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

def _init_metadata_table():
    """thread_metadata table banata hai agar exist nahi karta — thread name + PDF info store karta hai."""
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS thread_metadata (
            thread_id TEXT PRIMARY KEY,
            thread_name TEXT,
            filename TEXT,
            chunks INTEGER,
            pages INTEGER
        )
    """)
    conn.commit()

_init_metadata_table()

# -------------------
# 7. Graph
# -------------------
graph = StateGraph(ChatSchema)
graph.add_node('routing', routing)
graph.add_node('tools', tool_node)

graph.add_edge(START, 'routing')
graph.add_conditional_edges('routing', tools_condition)
graph.add_edge('tools', END)

workflow = graph.compile(checkpointer=checkpointer)

# -------------------
# 8. Helpers
# -------------------
def retrive_unique_thread():
    all_thread = set()
    for checkpoint in checkpointer.list(None):
        all_thread.add(checkpoint.config['configurable']['thread_id'])
    return list(all_thread)

def thread_has_document(thread_id: str) -> bool:
    """Check karta hai ke given thread_id ke liye FAISS retriever memory mein loaded hai ya nahi."""
    return str(thread_id) in _THREAD_RETRIEVERS

def thread_document_metadata(thread_id: str) -> dict:
    """DB se thread ki metadata (thread_name, filename, chunks, pages) nikalta hai. Persistent hai."""
    cursor = conn.cursor()
    cursor.execute(
        "SELECT thread_name, filename, chunks, pages FROM thread_metadata WHERE thread_id = ?",
        (str(thread_id),)
    )
    row = cursor.fetchone()
    if row is None:
        return {}
    return {
        "thread_name": row[0],
        "filename": row[1],
        "chunks": row[2],
        "pages": row[3],
    }

def save_thread_metadata(thread_id: str, metadata: dict):
    """DB mein thread ki metadata save/update karta hai (merge karta hai, overwrite nahi)."""
    existing = thread_document_metadata(str(thread_id))
    existing.update(metadata)

    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO thread_metadata (thread_id, thread_name, filename, chunks, pages)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(thread_id) DO UPDATE SET
            thread_name = excluded.thread_name,
            filename = excluded.filename,
            chunks = excluded.chunks,
            pages = excluded.pages
    """, (
        str(thread_id),
        existing.get("thread_name"),
        existing.get("filename"),
        existing.get("chunks"),
        existing.get("pages"),
    ))
    conn.commit()
    return existing

def ingest_pdf(file_bytes: bytes, thread_id: str, filename: Optional[str] = None) -> dict:
    """PDF ko read karke FAISS mein store karo (in-memory), aur metadata DB mein save karo (persistent)."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)

        vectorstore = FAISS.from_documents(chunks, embeddings)
        faiss_path = get_faiss_path(thread_id)

        vectorstore.save_local(faiss_path)

        retriever = vectorstore.as_retriever(
            search_kwargs={"k": 4}
        )

        _THREAD_RETRIEVERS[str(thread_id)] = retriever

        # ✅ Metadata DB mein save — thread_name preserve hota hai (merge logic save_thread_metadata mein hai)
        save_thread_metadata(str(thread_id), {
            "filename": filename or "uploaded.pdf",
            "chunks": len(chunks),
            "pages": len(docs)
        })

        os.unlink(tmp_path)
        return {"success": True, "chunks": len(chunks), "pages": len(docs)}

    except Exception as e:
        return {"success": False, "error": str(e)}

def remove_thread_document(thread_id: str):
    """Retriever memory se hatata hai, aur DB mein PDF-related fields clear karta hai (thread_name preserve hota hai)."""
    faiss_path = get_faiss_path(thread_id)

    if os.path.exists(faiss_path):
        shutil.rmtree(faiss_path)

    cursor = conn.cursor()
    cursor.execute("""
        UPDATE thread_metadata
        SET filename = NULL, chunks = NULL, pages = NULL
        WHERE thread_id = ?
    """, (str(thread_id),))
    conn.commit()

def clear_thread_checkpoint(thread_id: str):
    """Thread ki saari chat history (checkpoints + writes) DB se delete karta hai."""
    cursor = conn.cursor()
    cursor.execute("DELETE FROM checkpoints WHERE thread_id = ?", (str(thread_id),))
    cursor.execute("DELETE FROM writes WHERE thread_id = ?", (str(thread_id),))
    conn.commit()


def get_faiss_path(thread_id: str):
    return os.path.join(FAISS_DIR, str(thread_id))


    
def load_thread_retriever(thread_id: str):
    try:
        faiss_path = get_faiss_path(thread_id)

        if not os.path.exists(faiss_path):
            return None

        vectorstore = FAISS.load_local(
            faiss_path,
            embeddings,
            allow_dangerous_deserialization=True
        )

        retriever = vectorstore.as_retriever(
            search_kwargs={"k": 4}
        )

        _THREAD_RETRIEVERS[str(thread_id)] = retriever

        return retriever

    except Exception as e:
        print("FAISS Load Error:", e)
        return None