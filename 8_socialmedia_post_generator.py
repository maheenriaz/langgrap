from langchain_ollama import ChatOllama  # ← use this, not OllamaLLM
from pydantic import BaseModel, Field
from typing import TypedDict, Annotated,Literal
import operator
from langgraph.graph import StateGraph,START,END
import ollama
from langchain_core.messages import SystemMessage, HumanMessage

class TweetEvaluation(BaseModel):
    evaluation: Literal["approved", "needs_improvement"] = Field(..., description="Final evaluation result.")
    feedback: str = Field(..., description="feedback for the tweet.")

model = ChatOllama(model="llama3")

structured_evaluator_llm = model.with_structured_output(TweetEvaluation)
  
class SMSchema(TypedDict):
    topic:str
    tweet: str
    evaluation: Literal['approved','needs_improvement'] 
    feedback: str
    iteration: int
    tweet_history:dict
    max_iteration: int


def generate_post(state: SMSchema) -> SMSchema:
    prompt = f"""
    You are a funny and clever Twitter/X influencer.
    Write a short, original, and hilarious tweet on the topic: "{state['topic']}".

    Rules:
    - Do NOT use question-answer format.
    - Max 280 characters.
    - Use observational humor, irony, sarcasm, or cultural references.
    - Think in meme logic, punchlines, or relatable takes.
    - Use simple, day to day english
    """

    response = ollama.chat(
        model="qwen2.5:0.5b",
        messages=[{"role": "user", "content": prompt}]
    )
    state['tweet'] = response['message']['content']
    return state

def evaluate_post(state: SMSchema) -> SMSchema:
    prompt = f"""
    You are a ruthless, no-laugh-given Twitter critic.
    Evaluate the following tweet:

    Tweet: "{state['tweet']}"

    Criteria:
    - Originality
    - Humor
    - Punchiness
    - Virality Potential
    - Format

    Respond ONLY in structured format:
    - evaluation: "approved" or "needs_improvement"
    - feedback: One paragraph explaining strengths and weaknesses
    """

    output = structured_evaluator_llm.invoke(prompt)
    state['evaluation'] = output.evaluation
    state['feedback'] = output.feedback
    return state


def optimized_post(state: SMSchema) -> SMSchema:
    prompt = f"""
    Improve the tweet based on this feedback:
    "{state['feedback']}"

    Topic: "{state['topic']}"
    Original Tweet:
    {state['tweet']}

    Re-write it as a short, viral-worthy tweet. Avoid Q&A style and stay under 280 characters.
    """

    response = ollama.chat(
        model="qwen2.5:0.5b",
        messages=[{"role": "user", "content": prompt}]
    )
    new_tweet = response['message']['content']
    iteration = state['iteration'] + 1

    state['tweet'] = new_tweet
    state['iteration'] = iteration
    state['tweet_history'] = state.get('tweet_history', []) + [new_tweet]
    return state

def route_evaluation(state: SMSchema):

    if state['evaluation'] == 'approved' or state['iteration'] >= state['max_iteration']:
        return 'approved'
    else:
        return 'needs_improvement'
    
graph = StateGraph(SMSchema)

graph.add_node('generate_post',generate_post)
graph.add_node('evaluate_post',evaluate_post)
graph.add_node('optimized_post',optimized_post)


graph.add_edge(START,'generate_post')
graph.add_edge('generate_post','evaluate_post')
# iterative workflow
graph.add_conditional_edges('evaluate_post',route_evaluation,{'approved': END, 'needs_improvement':'optimized_post' })
graph.add_edge('optimized_post','evaluate_post')


workflow = graph.compile()

result = workflow.invoke({"topic":"indian railway", 'iteration':1,'max_iteration':3})
print(result)