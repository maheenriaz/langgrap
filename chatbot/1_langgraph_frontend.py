import streamlit as st
from langgraph_backend import workflow
from langchain_core.messages import SystemMessage,AIMessage, HumanMessage,BaseMessage

# with st.chat_message('user'):
#     st.text("hi")

# with st.chat_message('assistant'):
#     st.text("how can i help u")

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

message_history =[]

for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])

user_input = st.chat_input("type here")
config = {'configurable': {'thread_id': '1'}}
    
if user_input:
    st.session_state['message_history'].append({'role':'user', 'content':user_input})
    with st.chat_message('user'):
        st.text(user_input)

    result = workflow.invoke({'message':[HumanMessage(content=user_input)]},config=config)
    # print(result['message'][-1].content)
    st.session_state['message_history'].append({'role':'assistant', 'content':result['message'][-1].content})
    with st.chat_message('assistant'):
        st.text(result['message'][-1].content)