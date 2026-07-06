from dotenv import load_dotenv
load_dotenv()

import uuid
from typing import List
from pydantic import BaseModel, Field

from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_ollama import ChatOllama, OllamaEmbeddings  # pyright: ignore[reportMissingImports]
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.store.sqlite import SqliteStore
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.store.base import BaseStore   # <-- yeh line missing thi

# ----------------------------
# 2) System prompt
# ----------------------------
SYSTEM_PROMPT_TEMPLATE = """You are a helpful assistant with memory capabilities.
...
"""

# ----------------------------
# 3) Memory extraction LLM
# ----------------------------
memory_llm = ChatOllama(model="qwen2.5:7b")

class MemoryItem(BaseModel):
    text: str = Field(description="Atomic user memory")
    is_new: bool = Field(description="True if new, false if duplicate")

class MemoryDecision(BaseModel):
    should_write: bool
    memories: List[MemoryItem] = Field(default_factory=list)

# memory_extractor (LLM) se poochta hai ke kya naya memory‑worthy fact hai.
# Agar hai, to store.put() ke zariye SQLite LTM me save kar deta hai.
# Example: “I teach AI on YouTube” → stored as fact.
memory_extractor = memory_llm.with_structured_output(MemoryDecision)

MEMORY_PROMPT = """You are responsible for updating and maintaining accurate user memory.
...
"""

# ----------------------------
# 3) Nodes
# ----------------------------
# user ke latest message ko dekhta hai.
def remember_node(state: MessagesState, config: RunnableConfig, *, store: BaseStore):
    user_id = config["configurable"]["user_id"]
    ns = ("user", user_id, "details")

    items = store.search(ns)
    existing = "\n".join(it.value.get("data", "") for it in items) if items else "(empty)"

    last_text = state["messages"][-1].content

    decision: MemoryDecision = memory_extractor.invoke(
        [
            SystemMessage(content=MEMORY_PROMPT.format(user_details_content=existing)),
            {"role": "user", "content": last_text},
        ]
    )

    if decision.should_write:
        for mem in decision.memories:
            if mem.is_new and mem.text.strip():
                store.put(ns, str(uuid.uuid4()), {"data": mem.text.strip()})

    return {}

chat_llm = ChatOllama(model="qwen2.5:7b")


# chat_node user ke stored memories retrieve karta hai.
# System prompt me inject karta hai taake personalization ho.
# Phir ChatOllama se response generate hota hai.

def chat_node(state: MessagesState, config: RunnableConfig, *, store: BaseStore):
    user_id = config["configurable"]["user_id"]
    ns = ("user", user_id, "details")

    items = store.search(ns)
    user_details = "\n".join(it.value.get("data", "") for it in items) if items else ""

    system_msg = SystemMessage(
        content=SYSTEM_PROMPT_TEMPLATE.format(user_details_content=user_details or "(empty)")
    )

    response = chat_llm.invoke([system_msg] + state["messages"])
    return {"messages": [response]}

# ----------------------------
# 4) Build graph
# ----------------------------
builder = StateGraph(MessagesState)

builder.add_node("remember", remember_node)
builder.add_node("chat", chat_node)

builder.add_edge(START, "remember")
builder.add_edge("remember", "chat")
builder.add_edge("chat", END)

# ----------------------------
# 5) SQLite LTM + STM
# ----------------------------
# ----------------------------
# 5) SQLite Long-Term Memory + Short-Term Memory
# ----------------------------
with SqliteStore.from_conn_string("chatbot.db") as store, \
     SqliteSaver.from_conn_string("checkpoints.db") as checkpointer:

    store.setup()

    graph = builder.compile(
        store=store,
        checkpointer=checkpointer
    )

    config = {
        "configurable": {
            "user_id": "u1",
            "thread_id": "thread_1"
        }
    }

    graph.invoke({"messages": [{"role": "user", "content": "Hi, my name is Nitish"}]}, config)
    graph.invoke({"messages": [{"role": "user", "content": "I teach AI on YouTube"}]}, config)

    out = graph.invoke({"messages": [{"role": "user", "content": "Explain GenAI simply"}]}, config)
    print(out["messages"][-1].content)

    print("\n--- Stored Memories (from SQLite) ---")
    for it in store.search(("user", "u1", "details")):
        print(it.value["data"])
