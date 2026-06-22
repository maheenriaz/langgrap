from langgraph.graph import StateGraph,START,END
from typing import TypedDict
import ollama
import re


class SimpleLLMState(TypedDict):
    topic: str
    outline: str
    generated_text:str
    score: int


def generate_outline(state: SimpleLLMState)-> SimpleLLMState:
    prompt = f'Generate an short outline for a blog on the topic - {state['topic']}'
    response = ollama.chat(
        model="qwen2.5:0.5b",
        messages=[{"role": "user", "content": prompt}]
    )
    state['outline']= response['message']['content']
    return state

def generate_blog(state: SimpleLLMState)-> SimpleLLMState:
    getTitle = f'Generate a shortest 2 lines blog for this topic - {state['outline']}'
    response = ollama.chat(
        model="qwen2.5:0.5b",
        messages=[{"role": "user", "content": getTitle}]
    )
    state['generated_text']= response['message']['content']
    return state

def evaluate_blog(state: SimpleLLMState)-> SimpleLLMState:
    getScore = f'based on {state["outline"]}, rate my {state['generated_text']} and return only a single integer between 1 and 10.'
  
    response = ollama.chat(
        model="qwen2.5:0.5b",
        messages=[{"role": "user", "content": getScore}]
    )
    match = re.search(r'\d+', response['message']['content'])
    if match:
        state['score'] = int(match.group())
    else:
        state['score'] = 0
    # state['score']= response['message']['content']
    return state

graph = StateGraph(SimpleLLMState)

graph.add_node('generate_outline',generate_outline)
graph.add_node('generate_blog',generate_blog)
graph.add_node('evaluate_blog',evaluate_blog)


graph.add_edge(START,'generate_outline')
graph.add_edge('generate_outline','generate_blog')
graph.add_edge('generate_blog',"evaluate_blog")
graph.add_edge('evaluate_blog',END)

workflow = graph.compile()

result = workflow.invoke({'topic':"a cat is funniest and smart"})
print(result)
