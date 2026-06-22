from langchain_ollama import ChatOllama  # ← use this, not OllamaLLM
from pydantic import BaseModel, Field
from typing import TypedDict, Annotated,Literal
import operator
from langgraph.graph import StateGraph,START,END
import ollama

class SentimentSchema(BaseModel):
    feedback: Literal["Positive", "Negative"] = Field(
        description="Return only Positive or Negative"
    )

class DiagnosistSchema(BaseModel):
    issue_type: Literal["Spicy", "NonSpicy"] = Field(
        description="Classify if the review highlights a spicy or non-spicy issue"
    )
    tone: Literal["Formal", "Informal", "Neutral", "Empathetic"] = Field(
        description="Tone of the response"
    )
    urgency: Literal["Low", "Medium", "High"] = Field(
        description="Urgency level of the issue"
    )

model = ChatOllama(model="llama3")

structured_model = model.with_structured_output(SentimentSchema)
structured_model2 = model.with_structured_output(DiagnosistSchema)

# prompt = """""
# Analyze the sentiment of this text and return only 'Positive' or 'Negative' in the feedback field:
# Text: "this food is really yummy, i got happy after eating this at night."
# """""

# result = structured_model.invoke(prompt)
# print(result)
# print(result.feedback)


class SentimentState(TypedDict):
    review: str
    feedback: str
    result:str
    diagnosis: dict

def final_sentiment(state: SentimentState) -> SentimentState:
    prompt = f"Classify sentiment as Positive or Negative only:\n\n{state['review']}"
    output = structured_model.invoke(prompt)
    state['feedback'] = output.feedback
    state['result'] = output.feedback
    return state

def positive_response(state:SentimentState) -> SentimentState:
    prompt = f"""
    write a warm thank you message 2 lines shortly in the response to this review:
    
    {state["review"]} \n
    also, kindly ask user to leave feedback to our website.
    
    """

    response = ollama.chat(
        model="qwen2.5:0.5b",
        messages=[{"role": "user", "content": prompt}]
    )
    state['result']= response['message']['content']
    return state

def run_diagnosis(state:SentimentState) -> SentimentState:
    prompt = f"""
    Diagnose this negative review: \n\n {state['review']}
    return issue_type, tone and urgency
    """
    output = structured_model2.invoke(prompt)
    state['diagnosis'] = output.model_dump()
    return state
    # jo humny struc bnya tha issuetype,tone,urgency yeh aik dic ban k miljhyegi  
    # and diagosi state me bethjyegi

def negative_response(state:SentimentState) -> SentimentState:
    diagnosis = state['diagnosis']
    prompt = f"""
    you are a support analyst. the user had a {diagnosis['issue_type']}
    issue, sounded {diagnosis['tone']}, and marked urgency as {diagnosis['urgency']}
write and empatheic, helpful resilution message.
    {state["review"]} \n
    also, kindly ask user to leave feedback to our website.
    """
    response = ollama.chat(
        model="qwen2.5:0.5b",
        messages=[{"role": "user", "content": prompt}]
    )
    state['result']= response['message']['content']
    return state

def condition_check(state:SentimentState)-> Literal['positive_response','run_diagnosis']:
    if state['feedback'].lower() == "negative":
        return 'run_diagnosis'
    else:
        return 'positive_response'
    
graph = StateGraph(SentimentState)

graph.add_node('final_sentiment',final_sentiment)
graph.add_node('run_diagnosis',run_diagnosis)
graph.add_node('positive_response',positive_response)
graph.add_node('negative_response',negative_response)

# conditional workflow
graph.add_edge(START, 'final_sentiment' )
graph.add_conditional_edges("final_sentiment",condition_check)
graph.add_edge("run_diagnosis","negative_response")
graph.add_edge("positive_response",END)
graph.add_edge("negative_response",END)

workflow = graph.compile()

result = workflow.invoke({"review":"this food is really bad, i got vomit after eating this at night."})
print(result)


