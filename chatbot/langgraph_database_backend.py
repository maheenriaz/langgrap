from langchain_ollama import ChatOllama, OllamaEmbeddings # pyright: ignore[reportMissingImports]
from typing import TypedDict, Annotated, Optional, Dict, Any, List
from langgraph.graph import StateGraph, START, END # type: ignore
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage, BaseMessage # type: ignore
from langchain_core.runnables import RunnableConfig # type: ignore
from langgraph.graph.message import add_messages # type: ignore
from langgraph.checkpoint.sqlite import SqliteSaver # type: ignore
from langgraph.store.sqlite import SqliteStore # type: ignore
from langgraph.store.base import BaseStore # type: ignore
from langgraph.prebuilt import ToolNode, tools_condition # type: ignore
from langchain_community.tools import DuckDuckGoSearchRun # type: ignore
from langchain_community.document_loaders import PyPDFLoader # type: ignore
from langchain_community.vectorstores import FAISS # type: ignore
from langchain_text_splitters import RecursiveCharacterTextSplitter # type: ignore
import time

from langchain_core.tools import tool # type: ignore
from pydantic import BaseModel, Field # type: ignore
from dotenv import load_dotenv # type: ignore
import sqlite3
import requests # type: ignore
import os
import logging
import tempfile
import shutil
import uuid

FAISS_DIR = "faiss_indexes"
os.makedirs(FAISS_DIR, exist_ok=True)

GLOBAL_FAISS_PATH = FAISS_DIR  # ya full path: "/Users/owaisyameen/Desktop/langgraph/chatbot/faiss_indexes"
_GLOBAL_RETRIEVER = None

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

# Fixed single user_id - single-user app hai, isliye sab threads isi user ki memory share karenge
DEFAULT_USER_ID = "u1"

# -------------------
# 2. PDF Store (per thread) - retriever in-memory cache, FAISS disk pe persistent hai
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
    logging.info(f"[TOOL CALLED] Calculator -> {operation} on {first_num}, {second_num}")
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
    print(f"[TOOL CALLED] Stock Price -> {symbol}")
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
# 5. Long-Term Memory (LTM) - extraction schema + prompts
# -------------------

class MemoryItem(BaseModel):
    text: str = Field(description="Atomic user memory - ek chhota, self-contained fact ya preference")
    is_new: bool = Field(description="True agar yeh naya fact hai, False agar already stored memories mein maujood hai")

class MemoryDecision(BaseModel):
    should_write: bool = Field(description="True agar koi naya memory-worthy fact mila ho is message mein")
    memories: List[MemoryItem] = Field(default_factory=list)

memory_llm = ChatOllama(model="qwen2.5:7b")
memory_extractor = memory_llm.with_structured_output(MemoryDecision)

MEMORY_PROMPT = """You are responsible for maintaining accurate long-term memory about the user.

Look at the user's latest message and decide if it contains anything worth remembering long-term:
- Their name
- Preferences (likes/dislikes, communication style, format preferences)
- Decisions they've made in the past
- What has worked or not worked for them
- How they expect the assistant to behave for them
- Any other durable personal fact

Do NOT store one-off questions, greetings, or facts that are only relevant to the current message
(e.g. "what is the stock price of AAPL" is not memory-worthy).

Existing stored memories about this user:
{user_details_content}

If the new message contains a fact that is already captured above, mark is_new=False (or don't include it).
If it's genuinely new, mark is_new=True and write it as a short, standalone sentence.

Reply with should_write=True only if there's at least one new fact to save.
"""

SYSTEM_PROMPT_MEMORY_BLOCK = """
Known facts and preferences about this user (use these to personalize your response, but don't mention that you're using "stored memory" explicitly unless asked):
{user_details_content}
"""
# Yeh sirf saari memories ek simple list mein laata
# hai, taake UI (Streamlit) mein dikha sakein.
def _get_user_memory_text(store: BaseStore, user_id: str) -> str:
    """User ki saari stored memories ko ek text block mein jod kar deta hai."""
    ns = ("user", user_id, "details")
    items = store.search(ns)
    if not items:
        return "(empty)"
    return "\n".join(f"- {it.value.get('data', '')}" for it in items)

def extract_and_store_memory(store: BaseStore, user_id: str, last_text: str):
    """LLM se check karwata hai ke message mein koi naya memory-worthy fact hai, agar hai to store karta hai."""
    ns = ("user", user_id, "details")
    existing = _get_user_memory_text(store, user_id)

    try:
        decision: MemoryDecision = memory_extractor.invoke([
            SystemMessage(content=MEMORY_PROMPT.format(user_details_content=existing)),
            {"role": "user", "content": last_text},
        ])
    except Exception as e:
        logging.warning(f"[MEMORY] extraction failed: {e}")
        return

    if decision.should_write:
        for mem in decision.memories:
            if mem.is_new and mem.text.strip():
                store.put(ns, str(uuid.uuid4()), {"data": mem.text.strip()})
                logging.info(f"[MEMORY] saved: {mem.text.strip()}")

# -------------------
# 6. Nodes
# -------------------

# Yeh node har user message ke baad chalta hai. Iska kaam sirf yeh 
# hai: "dekho user ne kya bola, agar memory-worthy fact hai to save "
# "karo." Yeh khud koi reply nahi deta — isiliye return {} (khali)
# hai, koi message add nahi kar raha.

def remember_node(state: ChatSchema, config: RunnableConfig, *, store: BaseStore) -> ChatSchema:
    """Har user message ke baad chalta hai - automatic memory extraction."""
    user_id = config.get("configurable", {}).get("user_id", DEFAULT_USER_ID)
    last_text = state["messages"][-1].content

    if isinstance(last_text, str) and last_text.strip():
        extract_and_store_memory(store, user_id, last_text)

    return {}


# Yeh asal jawab dene wala node hai. Naya hissa yeh hai: 
# jawab dene se pehle, yeh user ki stored memories 
# nikalta hai aur unko system prompt (LLM ko diye gaye instructions)
# mein chipka deta hai. Isi liye LLM ko pata chal jata hai
# "is user ka naam Maheen hai, concise jawab pasand karta"
# " hai" — bina tumhe baar-baar batane ke.

def routing(state: ChatSchema, config: RunnableConfig = None, *, store: BaseStore) -> ChatSchema:
    thread_id = config.get("configurable", {}).get("thread_id")
    user_id = config.get("configurable", {}).get("user_id", DEFAULT_USER_ID)
    user_query = state["messages"][-1].content

    user_memory_text = _get_user_memory_text(store, user_id)
    memory_block = SYSTEM_PROMPT_MEMORY_BLOCK.format(user_details_content=user_memory_text)

    # --- Thread-specific PDF check (pehle) ---
    if thread_has_document(str(thread_id)):
        classify_prompt = f"""You are a query router. A PDF document is uploaded in this conversation.
Reply with ONLY one word — document or tool.
User question: {user_query}
Reply:"""
        intent = llm.invoke(classify_prompt).content.strip().lower()

        if "document" in intent:
            rag_result = rag_tool.invoke({"query": user_query, "thread_id": str(thread_id)})
            if "error" not in rag_result:
                context = "\n\n".join(rag_result["context"])
                response = llm.invoke(f"""You are a helpful assistant.
Answer ONLY from the provided PDF context.
{memory_block}
Context:
{context}
Question:
{user_query}""")
                return {"messages": [response]}

    # --- Global FAISS index check (dusra fallback) ---
    # --- Global FAISS index check (dusra fallback) ---
    if _GLOBAL_RETRIEVER is not None:
        # Pehle check karo ke query document se related hai ya nahi
        relevance_check = llm.invoke(
            f"""You are a query classifier for a banking assistant.
    Decide if the user's question is related to banking, finance, policies, loans, accounts, or any document/policy topic.
    Reply with ONLY one word: yes or no

    User question: {user_query}
    Reply:"""
        ).content.strip().lower()

        if "yes" in relevance_check:
            docs = _GLOBAL_RETRIEVER.invoke(user_query)
            if docs:
                context = "\n\n".join(doc.page_content for doc in docs)
                citations = []
                for doc in docs:
                    src = doc.metadata.get("source", "document")
                    page = doc.metadata.get("page", 0)
                    citations.append(f"{os.path.basename(src)}, page {int(page)+1}")
                citation_text = "\n".join(f"- {c}" for c in set(citations))

                response = llm.invoke(f"""You are a helpful banking assistant.
    Answer ONLY from the provided document context. If the answer is not in the context, say "I don't have this information in the available documents."
    Be concise and precise.
    {memory_block}

    Context:
    {context}

    Question:
    {user_query}

    At the end of your answer, add:
    Sources:
    {citation_text}""")
                print(f"[GLOBAL FAISS] Answered from global index for query: {user_query},{response}")
                return {"messages": [response]}
    
    
    
    
    
    
    # --- Tools fallback (web search, calculator, stock) ---
    system_message = SystemMessage(
                    content=f"""You are a helpful assistant.
            You can use calculator, web search, and stock price tool when needed.
            - Give concise, direct answers only.
            - Do NOT show raw tool outputs or JSON to the user.
            {memory_block}"""
                )
    response = llm_with_tools.invoke([system_message, *state["messages"]], config=config)
    return {"messages": [response]}



tool_node = ToolNode(tools)
# -------------------
# 7. Checkpointer + Store (long-term memory) + persistent metadata table
# -------------------
# IMPORTANT: checkpointer aur store dono apni khud ki transactions (BEGIN) chalate hain.
# Same connection share karna unsafe hai jab dono ek hi graph step mein use hon
# (transaction-within-transaction error). Isliye separate connections — same file,
# WAL mode + busy_timeout — taake reads/writes block na hon.

def _connect():
    c = sqlite3.connect(database='chatbot.db', check_same_thread=False, timeout=30,isolation_level=None,)
    c.execute("PRAGMA journal_mode=WAL;")
    c.execute("PRAGMA busy_timeout=30000;")
    return c

conn = _connect()
print("1")
checkpointer = SqliteSaver(conn=conn)
print("2")

store_conn = _connect()
print("3")

store = SqliteStore(conn=store_conn)
print("4")

store.setup()  # store ke internal tables banata hai (sirf pehli baar zaroori, baad mein no-op)
def load_global_faiss_index():
    """Existing FAISS index disk se load karta hai — sirf ek baar startup pe."""
    global _GLOBAL_RETRIEVER
    try:
        if not os.path.exists(os.path.join(GLOBAL_FAISS_PATH, "index.faiss")):
            logging.warning("[GLOBAL FAISS] index.faiss nahi mila, skipping.")
            return
        vectorstore = FAISS.load_local(
            GLOBAL_FAISS_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
        _GLOBAL_RETRIEVER = vectorstore.as_retriever(search_kwargs={"k": 4})
        logging.info("[GLOBAL FAISS] Index successfully loaded.")
    except Exception as e:
        logging.error(f"[GLOBAL FAISS] Load failed: {e}")


def _init_metadata_table():
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
print("5")
load_global_faiss_index()
# -------------------
# 8. Graph
# -------------------
graph = StateGraph(ChatSchema)
graph.add_node('remember', remember_node)
graph.add_node('routing', routing)
graph.add_node('tools', tool_node)

graph.add_edge(START, 'remember')
graph.add_edge('remember', 'routing')
graph.add_conditional_edges('routing', tools_condition)
graph.add_edge('tools', 'routing')

workflow = graph.compile(checkpointer=checkpointer, store=store)

# -------------------
# 9. Helpers
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
    """PDF ko read karke FAISS mein store karo (disk pe persistent), aur metadata DB mein save karo."""
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

        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        _THREAD_RETRIEVERS[str(thread_id)] = retriever

        # Metadata DB mein save - thread_name preserve hota hai (merge logic save_thread_metadata mein hai)
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
    """Retriever memory + disk se hatata hai, aur DB mein PDF-related fields clear karta hai (thread_name preserve hota hai)."""
    _THREAD_RETRIEVERS.pop(str(thread_id), None)

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

        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        _THREAD_RETRIEVERS[str(thread_id)] = retriever

        return retriever

    except Exception as e:
        print("FAISS Load Error:", e)
        return None

# -------------------
# 10. Long-Term Memory helpers (explicit access - UI ya manual "remember that..." commands ke liye)
# -------------------

def remember_fact_explicitly(text: str, user_id: str = DEFAULT_USER_ID):
    """User explicitly 'remember that I prefer X' jaisa bole, isko seedha call kar ke save kar sakte ho."""
    ns = ("user", user_id, "details")
    if text and text.strip():
        store.put(ns, str(uuid.uuid4()), {"data": text.strip()})

def get_all_user_memories(user_id: str = DEFAULT_USER_ID) -> List[Dict[str, str]]:
    """User ki saari stored memories list mein wapis deta hai (key + text) - sidebar mein dikhane/delete karne ke liye."""
    ns = ("user", user_id, "details")
    items = store.search(ns)
    if not items:
        return []
    return [{"key": it.key, "text": it.value.get("data", "")} for it in items]
# Ek specific memory delete karta hai — uski key (ID) se.
def delete_user_memory(memory_key: str, user_id: str = DEFAULT_USER_ID):
    """Ek specific memory item delete karta hai (key wahi uuid hai jo store.put mein use hua tha)."""
    ns = ("user", user_id, "details")
    store.delete(ns, memory_key)

def clear_all_user_memories(user_id: str = DEFAULT_USER_ID):
    """User ki saari long-term memories delete kar deta hai."""
    ns = ("user", user_id, "details")
    items = store.search(ns)
    for it in items:
        store.delete(ns, it.key)