from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
from typing import TypedDict, Annotated, Literal
import operator
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage, BaseMessage
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

class ChatSchema(TypedDict):
    message: Annotated[list[BaseMessage], add_messages]

llm = ChatOllama(model="qwen2.5:0.5b")

def generate_chat(state: ChatSchema) -> ChatSchema:
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        *state["message"]
    ]
    response = llm.invoke(messages)
    return {"message": [response]}

checkpointer = MemorySaver()
graph = StateGraph(ChatSchema)
graph.add_node('generate_chat', generate_chat)
graph.add_edge(START, 'generate_chat')
graph.add_edge('generate_chat', END)

workflow = graph.compile(checkpointer=checkpointer)
