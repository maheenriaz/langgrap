from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from langgraph.checkpoint.memory import InMemorySaver
import ollama

class JokeState(TypedDict):
    topic: str
    joke: str
    explanation: str

def generate_joke(state: JokeState):
    prompt = f'generate a joke on the topic {state["topic"]}'
    response = ollama.chat(
        model="qwen2.5:0.5b",
        messages=[{"role": "user", "content": prompt}]
    )
    return {'joke': response['message']['content']}

def generate_explanation(state: JokeState):
    prompt = f'write an explanation for the joke - {state["joke"]}'
    response = ollama.chat(
        model="qwen2.5:0.5b",
        messages=[{"role": "user", "content": prompt}]
    )
    return {'explanation': response['message']['content']}

# ✅ Define checkpointer once
checkpointer = InMemorySaver()

graph = StateGraph(JokeState)
graph.add_node('generate_joke', generate_joke)
graph.add_node('generate_explanation', generate_explanation)

graph.add_edge(START, 'generate_joke')
graph.add_edge('generate_joke', 'generate_explanation')
graph.add_edge('generate_explanation', END)

workflow = graph.compile(checkpointer=checkpointer)

# ✅ Define thread_id once
thread_id = "1"
config = {"configurable": {"thread_id": thread_id}}

# ✅ Reuse config everywhere
result = workflow.invoke({'topic': 'pizza'}, config=config)
print(result)
