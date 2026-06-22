from langchain_ollama import ChatOllama  # ← use this, not OllamaLLM
from pydantic import BaseModel, Field
from typing import TypedDict, Annotated
import operator
from langgraph.graph import StateGraph,START,END

class EvaluationSchema(BaseModel):
    feedback: str = Field(description="Detailed feedback the essay")
    score: int = Field(description="Score out of 10", ge=0, le=10)

model = ChatOllama(model="llama3")

structured_model = model.with_structured_output(EvaluationSchema)

prompt = """""
Pakistan is a country rich in culture and history.  
It was founded in 1947 as a homeland for Muslims of South Asia.  
The land is blessed with mountains, rivers, and fertile plains.  
Its people are resilient, hardworking, and full of spirit.  
Pakistan continues to strive for progress and unity.
"""""

result = structured_model.invoke(prompt)
print(result)
print(result.feedback)
print(result.score)

class UPSEssayState(TypedDict):
    essay:str
    language_feedback:str
    analysis_feedback:str
    clarity_feedback:str
    overall_feedback:str
    indivisual_score: Annotated[list[int],operator.add]
    avg_score:float

def evaluate_langgaue(state:UPSEssayState) -> UPSEssayState:
    prompt = f'evaluate the language quality of the following essay and provide a feedback and assign a score out of 10\n {state['essay']}'
    output = structured_model.invoke(prompt)
    return {'language_feedback' : output.feedback, 'indivisual_score':output.score}

def evaluate_analysis(state:UPSEssayState) -> UPSEssayState:
    prompt = f'evaluate the depth of analysis of the following essay and provide a feedback and assign a score out of 10\n {state['essay']}'
    output = structured_model.invoke(prompt)
    return {'analysis_feedback' : output.feedback, 'indivisual_score':output.score}

def evaluate_thought(state:UPSEssayState) -> UPSEssayState:
    prompt = f'evaluate the clarity of thought of the following essay and provide a feedback and assign a score out of 10\n {state['essay']}'
    output = structured_model.invoke(prompt)
    return {'clarity_feedback' : output.feedback, 'indivisual_score':output.score}

def evaluate_thought(state:UPSEssayState) -> UPSEssayState:
    prompt = f'evaluate the clarity of thought of the following essay and provide a feedback and assign a score out of 10\n {state['essay']}'
    output = structured_model.invoke(prompt)
    return {'clarity_feedback' : output.feedback, 'indivisual_score':output.score}

graph = StateGraph(UPSEssayState)


graph.add_node('evaluate_langgaue',evaluate_langgaue)
graph.add_node('evaluate_analysis',evaluate_analysis)
graph.add_node('evaluate_thought',evaluate_thought)
graph.add_node('final_evaluate',final_evaluate)

# parallel worflow 
graph.add_edge(START,'strike_rate_fun')
graph.add_edge(START,'run_boundary_fun')
graph.add_edge(START,"ball_per_boundary_fun")
graph.add_edge('strike_rate_fun','summary_func')
graph.add_edge('run_boundary_fun','summary_func')
graph.add_edge('ball_per_boundary_fun','summary_func')
graph.add_edge('summary_func',END)







