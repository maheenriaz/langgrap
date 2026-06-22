from langgraph.graph import StateGraph,START,END
from typing import TypedDict
import ollama
import re


class CricketLLMState(TypedDict):
    runs: int
    balls: int
    four:int
    six: int
    strike_rate:int
    running_boundary:int
    ball_per_boundary:int
    summary:str


def strike_rate_fun(state: CricketLLMState)-> CricketLLMState:
    get_strike_rate = (state['runs']/state['balls'])*100
    return {'strike_rate':get_strike_rate}

def run_boundary_fun(state: CricketLLMState)-> CricketLLMState:
    run_boundary_text = (4*state['four'])+(6*state['six'])
    return {'running_boundary':run_boundary_text}

def ball_per_boundary_fun(state: CricketLLMState)-> CricketLLMState:
    get_ball_per_boundary = state['balls']/(state['four'] +state['six'])
    return {'ball_per_boundary':get_ball_per_boundary}

def summary_func(state: CricketLLMState)-> CricketLLMState:
    summar_text = f"""
    strike rate - {state["strike_rate"]}
    ball per boundary - {state['ball_per_boundary']}
    running boundary - {state['running_boundary']}

    """
    return {'summary':summar_text}



graph = StateGraph(CricketLLMState)

graph.add_node('strike_rate_fun',strike_rate_fun)
graph.add_node('run_boundary_fun',run_boundary_fun)
graph.add_node('ball_per_boundary_fun',ball_per_boundary_fun)
graph.add_node('summary_func',summary_func)


graph.add_edge(START,'strike_rate_fun')
graph.add_edge(START,'run_boundary_fun')
graph.add_edge(START,"ball_per_boundary_fun")
graph.add_edge('strike_rate_fun','summary_func')
graph.add_edge('run_boundary_fun','summary_func')
graph.add_edge('ball_per_boundary_fun','summary_func')
graph.add_edge('summary_func',END)

workflow = graph.compile()

result = workflow.invoke({'runs':100,'balls':50,'four':6,'six':4})
print(result['summary'])
