# test_stream.py
from langgraph_backend import workflow
from langchain_core.messages import HumanMessage

config = {"configurable": {"thread_id": "test123"}}
stream = workflow.stream(
    {"message": [HumanMessage(content="hi")]},
    config=config,
    stream_mode="messages"
)

for chunk, metadata in stream:
    print(type(chunk), repr(chunk.content))