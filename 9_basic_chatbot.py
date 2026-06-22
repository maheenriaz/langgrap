from langchain_ollama import ChatOllama  # ← use this, not OllamaLLM
from pydantic import BaseModel, Field
from typing import TypedDict, Annotated,Literal
import operator
from langgraph.graph import StateGraph,START,END
import ollama
from langchain_core.messages import SystemMessage,AIMessage, HumanMessage,BaseMessage
from langgraph.graph.message import add_messages 
from langgraph.checkpoint.memory import MemorySaver #store data in memory/ram

class ChatSchema(TypedDict):
    message: Annotated[list[BaseMessage], add_messages] # type: ignore add_messages is reducer

def generate_chat(state: ChatSchema) -> ChatSchema:
    # Get full conversation history
    response = ollama.chat(
        model="qwen2.5:0.5b",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            *[
                {"role": "user" if isinstance(m, HumanMessage) else "assistant", "content": m.content}
                for m in state["message"]
            ]
        ]
    )
    output = response['message']['content']
    return {"message": [AIMessage(content=output)]}

checkpointer = MemorySaver()
graph = StateGraph(ChatSchema)

graph.add_node('generate_chat',generate_chat)

graph.add_edge(START,'generate_chat')
graph.add_edge('generate_chat',END)

workflow = graph.compile(checkpointer=checkpointer)

# 
thread_id= '1'
while True:
    user_message = input("type here:")
    if user_message.strip().lower() in ['exit','bye','quit']:
        break

    config = {'configurable': {'thread_id': thread_id}}
    result = workflow.invoke({'message':[HumanMessage(content=user_message)]},config=config)
    print(result['message'][-1].content)