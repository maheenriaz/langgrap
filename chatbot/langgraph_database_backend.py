from langchain_ollama import ChatOllama, OllamaEmbeddings # pyright: ignore[reportMissingImports]
from typing import TypedDict, Annotated, Optional, Dict, Any, List
from langgraph.graph import StateGraph, START, END # type: ignore
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage, BaseMessage, RemoveMessage # type: ignore
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
llm = ChatOllama(
    model="qwen2.5:7b",
    num_predict=256,     # max tokens limit karo (default kabhi bohot zyada hota hai)
    num_ctx=2048,        # context window chhota rakho agar zyada history use nahi ho rahi
    keep_alive="30m",
)

# NOTE: intent classification EK real LLM call hai (koi keyword/hardcoded rule nahi).
# 0.5b model choti/ambiguous queries pe kamzor tha, is liye thoda bara + deterministic
# (temperature=0) model use kar rahe hain taake classification consistent rahe.
llm_classifier = ChatOllama(
    model="qwen2.5:1.5b",
    temperature=0,        # deterministic classification - randomness nahi chahiye
    num_predict=10,        # sirf ek word chahiye, isliye chhota rakha
    num_ctx=2048,
    keep_alive="30m",
)

embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Chhota/cheap model sirf conversation history ko summarize karne ke liye -
# isse main llm ka context (num_ctx=2048) tools/RAG ke liye khali rehta hai.
llm_summarizer = ChatOllama(
    model="qwen2.5:0.5b",
    num_predict=300,
    num_ctx=2048,
    keep_alive="30m",
)

# Fixed single user_id - single-user app hai, isliye sab threads isi user ki memory share karenge
DEFAULT_USER_ID = "u1"

# -------------------
# 2. PDF Store (per thread) - retriever in-memory cache, FAISS disk pe persistent hai
# -------------------
_THREAD_RETRIEVERS: Dict[str, Any] = {}

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

tools = [search_tool, get_stock_price, calculator]
llm_with_tools = llm.bind_tools(tools)

# -------------------
# 4. State
# -------------------
class ChatSchema(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    summary: str  # older conversation ka running condensed summary (context compression)

# -------------------
# 5. Long-Term Memory (LTM) - extraction schema + prompts
# -------------------

class MemoryItem(BaseModel):
    text: str = Field(description="Atomic user memory - ek chhota, self-contained fact ya preference")
    is_new: bool = Field(description="True agar yeh naya fact hai, False agar already stored memories mein maujood hai")

class MemoryDecision(BaseModel):
    should_write: bool = Field(description="True agar koi naya memory-worthy fact mila ho is message mein")
    memories: List[MemoryItem] = Field(default_factory=list)

memory_llm = ChatOllama(model="qwen2.5:3b")   # ya llama3.1, mistral, etc.
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

def _get_user_memory_text(store: BaseStore, user_id: str) -> str:
    """User ki saari stored memories ko ek text block mein jod kar deta hai."""
    ns = ("user", user_id, "details")
    items = store.search(ns)
    if not items:
        return "(empty)"
    return "\n".join(f"- {it.value.get('data', '')}" for it in items)

def extract_and_store_memory(store: BaseStore, user_id: str, last_text: str):
    ns = ("user", user_id, "details")
    existing = _get_user_memory_text(store, user_id)

    try:
        decision: MemoryDecision = memory_extractor.invoke([
            SystemMessage(content=MEMORY_PROMPT.format(user_details_content=existing)),
            {"role": "user", "content": last_text},
        ], config={"tags": ["memory_extractor"]})
        print(f"[MEMORY DEBUG] decision = {decision}")
    except Exception as e:
        logging.warning(f"[MEMORY] extraction failed: {e}")
        return

    if decision.should_write:
        for mem in decision.memories:
            print(f"[MEMORY DEBUG] mem = {mem}")
            if mem.is_new and mem.text.strip():
                store.put(ns, str(uuid.uuid4()), {"data": mem.text.strip()})
                logging.info(f"[MEMORY] saved: {mem.text.strip()}")
    else:
        print("[MEMORY DEBUG] should_write=False, kuch save nahi hua")

# -------------------
# 6. Nodes
# -------------------

def remember_node(state: ChatSchema, config: RunnableConfig, *, store: BaseStore) -> ChatSchema:
    print("[REMEMBER NODE] called")
    user_id = config.get("configurable", {}).get("user_id", DEFAULT_USER_ID)
    last_text = state["messages"][-1].content
    print(f"[REMEMBER NODE] last_text = {repr(last_text)}")

    if isinstance(last_text, str) and len(last_text.strip()) >= 4:
        extract_and_store_memory(store, user_id, last_text)

    return {}

# -------------------
# 6b. Context Compression (running summary of old turns)
# -------------------
# Jab total messages is threshold se zyada ho jayen, purani messages ko
# summarize kar ke ek chhoti si "summary" mein fold kar dete hain, aur
# sirf recent KEEP_RECENT_MESSAGES messages verbatim rakhte hain. Isse
# num_ctx=2048 wale chhote local models pe bhi lambi chat chalti rehti hai.
MAX_MESSAGES_BEFORE_COMPRESSION = 12
KEEP_RECENT_MESSAGES = 6

SUMMARY_PROMPT = """You are compressing an ongoing chat conversation so it fits in a small context window.

Existing summary of everything before these messages:
{existing_summary}

New messages to fold into that summary (oldest to newest):
{new_messages_text}

Write ONE updated summary (max ~150 words) that:
- Keeps important facts, user requests, decisions, and any numbers/names mentioned
- Keeps unresolved questions or pending tasks
- Drops small talk, greetings, and anything not needed to continue the conversation
- Is written in plain prose, no bullet points, no preamble

Reply with ONLY the updated summary text, nothing else.
"""


def _messages_to_text(messages: list[BaseMessage]) -> str:
    lines = []
    for m in messages:
        content = m.content if isinstance(m.content, str) else str(m.content)
        content = content.strip()
        if content:
            lines.append(f"{m.type}: {content}")
    return "\n".join(lines)


def compress_node(state: ChatSchema, config: RunnableConfig = None) -> ChatSchema:
    """Purani messages ko summary mein compress karta hai jab list bohot lambi ho jaye."""
    messages = state["messages"]

    if len(messages) <= MAX_MESSAGES_BEFORE_COMPRESSION:
        return {}

    existing_summary = state.get("summary") or "(no summary yet)"
    to_summarize = messages[:-KEEP_RECENT_MESSAGES]
    convo_text = _messages_to_text(to_summarize)

    if not convo_text:
        return {}

    try:
        # tag zaroor lagao - is call ke tokens frontend stream mein user ko kabhi
        # nahi dikhne chahiye, tag ke bina wo "routing"/"compress" node ke andar
        # kisi bhi other stream ke sath mix ho kar UI mein leak ho sakte hain.
        response = llm_summarizer.invoke(
            SUMMARY_PROMPT.format(existing_summary=existing_summary, new_messages_text=convo_text),
            config={"tags": ["summarizer"]},
        )
        new_summary = response.content.strip()
    except Exception as e:
        logging.warning(f"[COMPRESS] summarization failed, keeping messages as-is: {e}")
        return {}

    if not new_summary:
        return {}

    # add_messages reducer RemoveMessage(id=...) ko dekh kar us id wali message
    # state se hata deta hai - isse purani messages checkpoint se bhi effectively hat jati hain.
    remove_ops = [RemoveMessage(id=m.id) for m in to_summarize if getattr(m, "id", None) is not None]

    logging.info(
        f"[COMPRESS] folded {len(to_summarize)} old messages into summary, "
        f"kept last {len(messages) - len(to_summarize)} verbatim"
    )

    return {"messages": remove_ops, "summary": new_summary}


# -------------------
# 6c. Retrieved-Context Compression (RAG chunks: metadata headers + page_content)
# -------------------
# Har retrieved chunk apne headers (H1/H2/H3) + full page_content ke sath aata hai -
# 5 chunks milte hi yeh text bohot bara ho sakta hai (num_ctx=2048 ke liye).
#
# IMPORTANT: yeh EXTRACTIVE compression hai - koi LLM paraphrasing/rewriting nahi.
# Pehle try kiya tha ke ek chhota model (0.5b) is context ko summarize kare, lekin
# chhote models factual/policy content mein hallucinate kar dete hain (naye "facts"
# bana dete hain jo source docs mein thay hi nahi - jaisa aap ne dekha). Is liye
# ab hum sirf: (1) sabse relevant chunks select karte hain (FAISS score se already
# sorted), (2) har chunk ko character-budget tak truncate karte hain - text verbatim
# rehta hai, kabhi rewrite nahi hota, is se hallucination possible hi nahi.
MAX_CONTEXT_CHARS = 3000       # total context ka overall budget
MAX_CHARS_PER_CHUNK = 800      # ek chunk ka max size (isse zyada bara ho to truncate)


def _format_chunk(doc) -> str:
    headers = []
    if "H1" in doc.metadata:
        headers.append(doc.metadata["H1"])
    if "H2" in doc.metadata:
        headers.append(doc.metadata["H2"])
    if "H3" in doc.metadata:
        headers.append(doc.metadata["H3"])

    content = doc.page_content.strip()
    if len(content) > MAX_CHARS_PER_CHUNK:
        # word boundary pe truncate karo, taake beech mein word na kate
        content = content[:MAX_CHARS_PER_CHUNK].rsplit(" ", 1)[0] + " ..."

    return f"Section: {' > '.join(headers)}\n{content}\n\n"


def compress_retrieved_context(relevant_docs: list) -> str:
    """
    Docs already relevance-order mein hain (FAISS score ascending = most similar first).
    Budget khatam hote hi baaki chunks drop kar dete hain - koi paraphrasing nahi,
    is liye source content se koi deviation/hallucination nahi ho sakta.
    """
    pieces = []
    total_len = 0

    for doc in relevant_docs:
        piece = _format_chunk(doc)
        if pieces and (total_len + len(piece) > MAX_CONTEXT_CHARS):
            break
        pieces.append(piece)
        total_len += len(piece)

    compressed = "".join(pieces)
    logging.info(
        f"[CONTEXT COMPRESS] {len(relevant_docs)} candidate chunks -> "
        f"{len(pieces)} kept, {total_len} chars (extractive, no LLM rewrite)"
    )
    return compressed


# -------------------
# 6d. Semantic Cache (skip retrieval + LLM call for near-duplicate questions)
# -------------------
# Agar koi question pehle bhi (ya usi jaisa, semantically) poocha ja chuka hai,
# to seedha stored answer return kar dete hain - FAISS retrieval aur LLM call
# dono skip ho jate hain. IMPORTANT: yeh sirf RAG/policy-answer path ke liye use
# hota hai (deterministic Q&A). Tool-calling path (stock price, web search,
# calculator) ko KABHI cache mat karo - wahan answers time-sensitive/dynamic
# hote hain aur stale result dena galat hoga.
SEMANTIC_CACHE_DIR = "semantic_cache_index"
os.makedirs(SEMANTIC_CACHE_DIR, exist_ok=True)

# FAISS L2 distance hai - jitna kam utna zyada similar (0 = identical text).
# Yeh threshold aapke embedding model/data ke hisaab se tune karna hoga.
# Pehle console mein "[SEMANTIC CACHE] best match score=..." logs dekh lein
# (jaisa RAG retrieval ke docs_with_scores print hote hain), phir yahan
# ek sensible cutoff set karein.
SEMANTIC_CACHE_SCORE_THRESHOLD = 0.05

_SEMANTIC_CACHE_STORE = None
_semantic_cache_load_attempted = False


def _load_semantic_cache():
    """Disk se existing semantic cache index load karta hai (agar mojood ho)."""
    global _SEMANTIC_CACHE_STORE, _semantic_cache_load_attempted
    _semantic_cache_load_attempted = True
    try:
        if os.path.exists(os.path.join(SEMANTIC_CACHE_DIR, "index.faiss")):
            _SEMANTIC_CACHE_STORE = FAISS.load_local(
                SEMANTIC_CACHE_DIR,
                embeddings,
                allow_dangerous_deserialization=True,
            )
            logging.info("[SEMANTIC CACHE] existing cache index loaded from disk.")
    except Exception as e:
        logging.warning(f"[SEMANTIC CACHE] load failed, starting fresh: {e}")
        _SEMANTIC_CACHE_STORE = None


def get_cached_answer(query: str) -> Optional[str]:
    """Agar semantically similar question pehle poocha ja chuka ho, uska stored answer deta hai, warna None."""
    global _SEMANTIC_CACHE_STORE
    if not _semantic_cache_load_attempted:
        _load_semantic_cache()

    if _SEMANTIC_CACHE_STORE is None:
        return None
    print(f"_SEMANTIC_CACHE_STORE = {_SEMANTIC_CACHE_STORE}")
    try:
        results = _SEMANTIC_CACHE_STORE.similarity_search_with_score(query, k=1)
    except Exception as e:
        logging.warning(f"[SEMANTIC CACHE] lookup failed: {e}")
        return None

    if not results:
        return None

    doc, score = results[0]
    logging.info(f"[SEMANTIC CACHE] best match score={score} for query={query!r}")

    if score <= SEMANTIC_CACHE_SCORE_THRESHOLD:
        return doc.metadata.get("answer")
    return None


def store_cached_answer(query: str, answer: str):
    """Naya (query -> answer) pair cache mein save karta hai aur disk pe persist karta hai."""
    global _SEMANTIC_CACHE_STORE
    if not answer or not answer.strip():
        return
    try:
        if _SEMANTIC_CACHE_STORE is None:
            _SEMANTIC_CACHE_STORE = FAISS.from_texts(
                [query], embeddings, metadatas=[{"answer": answer}]
            )
        else:
            _SEMANTIC_CACHE_STORE.add_texts([query], metadatas=[{"answer": answer}])
        _SEMANTIC_CACHE_STORE.save_local(SEMANTIC_CACHE_DIR)
    except Exception as e:
        logging.warning(f"[SEMANTIC CACHE] store failed: {e}")


def clear_semantic_cache():
    """Poora semantic cache disk + memory se hata deta hai - naya/updated policy doc upload hone par yeh call karo,
    warna purane docs ke stale cached answers milte rahenge."""
    global _SEMANTIC_CACHE_STORE
    _SEMANTIC_CACHE_STORE = None
    if os.path.exists(SEMANTIC_CACHE_DIR):
        shutil.rmtree(SEMANTIC_CACHE_DIR)
        os.makedirs(SEMANTIC_CACHE_DIR, exist_ok=True)


# -------------------
# 6e. Intent Classification (100% LLM-based - koi hardcoded keyword rule nahi)
# -------------------
# Purana prompt bohot generic tha aur 0.5b model chhoti/edge-case queries
# (jaise "resign after taking a loan") ko "general" misclassify kar raha tha.
# Fix: (1) thoda bara/deterministic model (temperature=0), (2) few-shot examples
# taake model ko pattern samajh aaye ke "employee benefit/entitlement/process se
# related koi bhi cheez jo company document mein defined hoti hai" wo policy hai,
# (3) strict output parsing.
INTENT_PROMPT = """You classify a user's message into exactly one category: policy or general.

"policy" = the question is about an HR policy, company rule, employee benefit,
entitlement, procedure, or anything whose answer would come from an official
HR/company document (leave, loans/financing, notice period, resignation process,
salary, insurance, bonus, termination, attendance, code of conduct, etc.)

"general" = greetings, small talk, the user sharing personal info (like their name),
general knowledge questions, math, stock prices, or anything NOT requiring a
company document lookup.

Examples:
Message: What is the maternity leave policy?
Answer: policy

Message: An employee is resigning after taking a personal loan. What notice period must they serve, and what happens to their outstanding staff financing?
Answer: policy

Message: Hi, how are you?
Answer: general

Message: What is 25 * 4?
Answer: general

Message: My name is Ali, remember that.
Answer: general

Message: How many sick leaves am I entitled to per year?
Answer: policy

Message: An employee wants to apply for house building finance. Are they eligible after two years of service, and what is the maximum repayment period?
Answer: policy

Now classify this message. Reply with ONLY one word - either "policy" or "general" - nothing else.

Message: {query}
Answer:"""


def _classify_intent(query: str) -> str:
    """LLM se query classify karwata hai. Koi keyword/hardcoded rule involved nahi -
    agar LLM call fail ho jaye to hi safe default 'general' use hota hai."""
    try:
        resp = llm_classifier.invoke(
            INTENT_PROMPT.format(query=query),
            config={"tags": ["intent_classifier"]},
        )
        label = resp.content.strip().lower()
        # strict parsing - label ke andar "policy" word dhoondo (model kabhi
        # "Answer: policy" jaisa likh sakta hai chhote token budget ke bawajood)
        if "policy" in label:
            return "policy"
        if "general" in label:
            return "general"
        # ambiguous/garbage output - safe default
        logging.warning(f"[INTENT] unrecognized classifier output: {label!r}, defaulting to general")
        return "general"
    except Exception as e:
        logging.warning(f"[INTENT] classification failed, defaulting to general: {e}")
        return "general"


# -------------------
# 6f. Retrieval-confidence fallback (NOT a keyword rule)
# -------------------
# Chhota classifier model naye/anjaane phrasing pe har baar galti kar sakta hai -
# few-shot examples add karte rehna "whack-a-mole" hai, kyunki naya wording aata
# rahega. Isliye jab classifier "general" bole, hum ek extra check karte hain:
# seedha FAISS policy index se puchte hain ke is query ka koi STRONG match
# maujood hai ya nahi. Agar haan, to classifier ke faisle ko override kar dete
# hain. Yeh koi keyword list nahi hai - yeh actual indexed policy document se
# ek data-driven confidence signal hai, jo classifier se zyada reliable hai
# kyunki wo asli document content se match kar raha hai, na ke sirf wording se.
#
# TUNING: console mein "[RETRIEVAL FALLBACK] best_score=..." dekh kar is
# threshold ko adjust karo, bilkul waise hi jaise SEMANTIC_CACHE_SCORE_THRESHOLD
# tune kiya gaya tha. FAISS L2 distance hai - jitna kam utna zyada similar.
RETRIEVAL_FALLBACK_SCORE_THRESHOLD = 0.9


def _has_strong_policy_match(query: str) -> bool:
    """GLOBAL_RETRIEVER ke FAISS index mein query ka koi confidently-close match
    hai ya nahi, ye check karta hai. Classifier ke 'general' faisle ko override
    karne ke liye use hota hai."""
    if _GLOBAL_RETRIEVER is None:
        return False
    try:
        docs_with_scores = _GLOBAL_RETRIEVER.vectorstore.similarity_search_with_score(query, k=1)
    except Exception as e:
        logging.warning(f"[RETRIEVAL FALLBACK] lookup failed: {e}")
        return False

    if not docs_with_scores:
        return False

    _, best_score = docs_with_scores[0]
    logging.info(f"[RETRIEVAL FALLBACK] best_score={best_score} for query={query!r}")
    return best_score <= RETRIEVAL_FALLBACK_SCORE_THRESHOLD


# -------------------
# 6g. Answer Confidence Score (FAISS similarity score se derive hota hai)
# -------------------
# RAG answer ke sath ek High/Medium/Low confidence badge dikhate hain, taake user
# ko pata chale ke answer kitna strongly policy document se grounded hai.
# FAISS L2 distance hai - jitna kam score utna zyada similar/confident match.
#
# TUNING: console mein already "[DEBUG] scores = [...]" print ho raha hai -
# wahan actual scores dekh kar ye thresholds tune karo (bilkul waise hi jaise
# baaki thresholds is file mein tune kiye gaye hain).
CONFIDENCE_HIGH_THRESHOLD = 0.5      # is se kam/equal score => High
CONFIDENCE_MEDIUM_THRESHOLD = 0.9    # is se kam/equal (but > high) => Medium, warna Low

# Marker jo answer text ke end mein chupa kar bheja jata hai - frontend isko
# parse kar ke ek colored badge dikhata hai aur marker khud text se hata deta hai.
CONFIDENCE_MARKER_TEMPLATE = "\n\n[[CONFIDENCE::{label}]]"


def _confidence_label(best_score: float) -> str:
    """Best (sabse chhota/similar) FAISS score se HIGH/MEDIUM/LOW label banata hai."""
    if best_score <= CONFIDENCE_HIGH_THRESHOLD:
        return "HIGH"
    elif best_score <= CONFIDENCE_MEDIUM_THRESHOLD:
        return "MEDIUM"
    else:
        return "LOW"


def routing(state: ChatSchema, config: RunnableConfig = None, *, store: BaseStore, skip_cache: bool = False) -> ChatSchema:
    thread_id = config.get("configurable", {}).get("thread_id")
    user_id = config.get("configurable", {}).get("user_id", DEFAULT_USER_ID)
    user_query = state["messages"][-1].content

    user_memory_text = _get_user_memory_text(store, user_id)
    memory_block = SYSTEM_PROMPT_MEMORY_BLOCK.format(user_details_content=user_memory_text)

    conversation_summary = state.get("summary")
    if conversation_summary:
        memory_block += f"\nSummary of earlier parts of this conversation (older messages were compressed to save context):\n{conversation_summary}\n"

    if _GLOBAL_RETRIEVER is None:
        intent = "general"
    else:
        llm_intent = _classify_intent(user_query)
        if llm_intent == "policy":
            intent = "policy"
        elif _has_strong_policy_match(user_query):
            # LLM ne "general" bola tha, lekin policy index mein strong match mila -
            # classifier ki galti ko yahan correct kar rahe hain (data-driven, koi
            # keyword hardcode nahi).
            logging.info("[ROUTING] LLM classified as general but retrieval found a strong match -> overriding to policy")
            intent = "policy"
        else:
            intent = "general"
    use_rag = intent == "policy"

    print("user_query = ", user_query, " | intent =", intent)
    logging.info(f"[ROUTING] intent={intent}")

    if use_rag:
        # skip_cache=True (regenerate ke waqt frontend se aata hai) -> cache lookup
        # bypass karo, warna regenerate button hamesha wohi purana cached answer
        # wapis de deta jo pehle se stored hai.
        cached_answer = None if skip_cache else get_cached_answer(user_query)
        if cached_answer is not None:
            logging.info("[SEMANTIC CACHE] hit - skipping retrieval + LLM call")
            return {"messages": [AIMessage(content=cached_answer)]}

        vectorstore = _GLOBAL_RETRIEVER.vectorstore
        docs_with_scores = vectorstore.similarity_search_with_score(user_query, k=10)
        relevant_docs = [doc for doc, score in docs_with_scores[:5]]

        for doc, score in docs_with_scores:
            print("=" * 80)
            print(score)
            print(doc.metadata)
            print(doc.page_content)

        print(f"[DEBUG] scores = {[s for _, s in docs_with_scores]}")
        print(f"[DEBUG] {len(relevant_docs)} relevant docs selected for context")

        if relevant_docs:
            context = compress_retrieved_context(relevant_docs)
            print("\n===== CONTEXT SENT TO LLM =====")
            print(context)
            print("===================")
            prompt = f"""
                You are an HR policy assistant.

                Rules:
                1. Answer ONLY from the provided context.
                2. If the answer can be directly inferred from the context, answer it.
                3. Do NOT say information is unavailable if the context clearly contains the answer.
                4. Do NOT use outside knowledge.
                5. The context may include multiple retrieved sections - some may NOT be relevant
                   to the question. Ignore any section that isn't relevant; do not mention or
                   summarize unrelated sections in your answer.

                {memory_block}
                Context:
                {context}

                Question:
                {user_query}

                Answer:
                """
            response = llm.invoke(prompt)

            # docs_with_scores pehle se relevance-order (ascending = most similar
            # first) mein hai, isliye [0] hi best/sabse confident match hai.
            best_score_for_confidence = docs_with_scores[0][1]
            confidence_label = _confidence_label(best_score_for_confidence)
            logging.info(f"[CONFIDENCE] best_score={best_score_for_confidence} -> {confidence_label}")

            final_content = response.content.strip() + CONFIDENCE_MARKER_TEMPLATE.format(label=confidence_label)

            store_cached_answer(user_query, final_content)
            return {"messages": [AIMessage(content=final_content)]}

        # relevant_docs khali the - fallback general answer taake response undefined na ho
        intent = "general"

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

store.setup()
def load_global_faiss_index():
    """Existing FAISS index disk se load karta hai — sirf ek baar startup pe."""
    global _GLOBAL_RETRIEVER
    try:
        if not os.path.exists(os.path.join(GLOBAL_FAISS_PATH, "index.faiss")):
            print("Looking for FAISS at:", os.path.abspath(GLOBAL_FAISS_PATH))
            logging.warning("[GLOBAL FAISS] index.faiss nahi mila, skipping.")
            return
        vectorstore = FAISS.load_local(
            GLOBAL_FAISS_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
        _GLOBAL_RETRIEVER = vectorstore.as_retriever(search_kwargs={"k": 10})
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
graph.add_node('compress', compress_node)

graph.add_edge(START, 'remember')
graph.add_edge('remember', 'routing')
# tools_condition normally routes straight to END when no tool call is needed -
# route it through 'compress' instead so old messages get folded into the
# summary before the turn finishes and gets checkpointed.
graph.add_conditional_edges('routing', tools_condition, {"tools": "tools", END: "compress"})
graph.add_edge('tools', 'routing')
graph.add_edge('compress', END)

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