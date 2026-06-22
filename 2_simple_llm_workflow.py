from langgraph.graph import StateGraph,START,END
from typing import TypedDict
import ollama

class SimpleLLMState(TypedDict):
    question: str
    answer: str


def llm_qa(state: SimpleLLMState)-> SimpleLLMState:
    response = ollama.chat(
        model="llama3",
        messages=[{"role": "user", "content": state['question']}]
    )
    state['answer']= response['message']['content']
    return state

graph = StateGraph(SimpleLLMState)

graph.add_node('llm_qa',llm_qa)

graph.add_edge(START,'llm_qa')
graph.add_edge('llm_qa',END)

workflow = graph.compile()

result = workflow.invoke({'question':"hi"})
print(result)
