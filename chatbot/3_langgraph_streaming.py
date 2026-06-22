import ollama
import streamlit as st
from langgraph_backend import workflow
from langchain_core.messages import HumanMessage

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

# Show old messages
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.text(message["content"])

user_input = st.chat_input("type here")
config = {"configurable": {"thread_id": "1"}}

if user_input:
    # Add user message
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.text(user_input)

    # Placeholder for streaming output
    with st.chat_message("assistant"):
        placeholder = st.empty()
        streamed_text = ""

        response = ollama.chat(
            model="qwen2.5:0.5b",
            messages=[{"role": "user", "content": user_input}],
            stream=True
        )

        for chunk in response:
            if "message" in chunk and "content" in chunk["message"]:
                token = chunk["message"]["content"]
                streamed_text += token
                placeholder.text(streamed_text)   # live typing effect

        st.session_state["message_history"].append({"role": "assistant", "content": streamed_text})
