from langgraph.graph import StateGraph,START,END
from typing import TypedDict

# define state 
class BMIState(TypedDict):
    weight_kg: float
    height_m:float
    bmi:float
    overWeight: bool


def caculate_bmi(state: BMIState) -> BMIState:
    weight = state['weight_kg']
    height = state['height_m']

    bmi = weight/(height**2)

    state['bmi'] =bmi

    return state

def label_bmi(state:BMIState)-> BMIState:
    if(state['bmi'] > 20):
        state['overWeight'] = True
    else:
        state['overWeight'] = False
    return state
        
# define graphh 
graph = StateGraph(BMIState)

# add nodes 
graph.add_node('calculate_bmi',caculate_bmi)
graph.add_node('label_bmi',label_bmi)
# add edges 
graph.add_edge(START, 'calculate_bmi')
graph.add_edge('calculate_bmi', 'label_bmi')
graph.add_edge("label_bmi",END)
# compile graph 
workflow = graph.compile()
# execute the graph 
result = workflow.invoke({"weight_kg": 54, "height_m": 1.75})
print(result)


# class StudentResultState(TypedDict):
#     marksObtained: int
#     totalMarks: int
#     studentGrade: str
#     finalPercentage: int

# def calculate_percentage(state: StudentResultState) -> StudentResultState:
#     percentage = state['marksObtained']/state['totalMarks']
#     state['finalPercentage'] = percentage

#     return state

# def assign_grade(state:StudentResultState) -> StudentResultState:
#     if(state['marksObtained'] > 85):
#         state['studentGrade'] = 'A'
#     elif(state['marksObtained'] > 75):
#         state['studentGrade'] = 'B'
#     else:
#         state['studentGrade'] = 'C'

#     return state

# graph = StateGraph(StudentResultState)

# graph.add_node("calculate_percentage",calculate_percentage)
# graph.add_node("assign_grade",assign_grade)

# graph.add_edge(START,'calculate_percentage')
# graph.add_edge('calculate_percentage','assign_grade')
# graph.add_edge('assign_grade',END)

# workflow = graph.compile()
# result = workflow.invoke({'marksObtained':55,'totalMarks':100})
# print(result)
