from langchain_ollama import ChatOllama  # ← use this, not OllamaLLM
from pydantic import BaseModel, Field
from typing import TypedDict, Annotated,Literal
import operator
from langgraph.graph import StateGraph,START,END


class EquationState(TypedDict):
    a:int
    b:int
    c:int

    equation:str
    discriminant: float
    result: str

def show_equation(state: EquationState) -> EquationState:
    equation = f"{state['a']}x2-{state['b']}-x{state['c']}"
    return {'equation': equation}

def calculate_discriminant(state: EquationState) -> EquationState:
    discriminant_value = (state['b']**2) - 4*(state['a'] * state['c'])
    return {'discriminant': discriminant_value}

def real_root(state: EquationState) -> EquationState:
    return {'result':"real_root"}

def repeated_root(state: EquationState) -> EquationState:
    return {'result':"repeated_root"}

def non_real_roots(state: EquationState) -> EquationState:
    return {'result':"non_real_roots"}

def check_condition(state: EquationState) -> Literal["real_root","repeated_root","non_real_roots"]:
    if(state['discriminant'] > 0):
        return "real_root"
    elif(state['discriminant'] < 0):
        return "non_real_roots"
    else:
        return "repeated_root"
    

graph = StateGraph(EquationState)

graph.add_node('show_equation',show_equation)
graph.add_node('calculate_discriminant',calculate_discriminant)
graph.add_node('real_root',real_root)
graph.add_node('repeated_root',repeated_root)
graph.add_node('non_real_roots',non_real_roots)

# conditional workflow
graph.add_edge(START,'show_equation')
graph.add_edge('show_equation','calculate_discriminant')
graph.add_conditional_edges('calculate_discriminant',check_condition)
graph.add_edge('real_root',END)
graph.add_edge('repeated_root',END)
graph.add_edge('non_real_roots',END)
# graph.add_edge(START,'show_equation')
# graph.add_edge(START,'show_equation')

workflow = graph.compile()

result = workflow.invoke({'a':1,'b':2,'c':3})
print(result)