import logging
import streamlit as st
from langgraph_database_backend import (
    workflow, retrive_unique_thread,
    ingest_pdf, thread_has_document, thread_document_metadata,
    save_thread_metadata, remove_thread_document,
    clear_thread_checkpoint
)
from langchain_core.messages import HumanMessage
import uuid

def generate_thread_id():
    return uuid.uuid4()

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    add_thread(thread_id)
    st.session_state['message_history'] = []

def add_thread(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)

def load_conversations(thread_id):
    state = workflow.get_state({"configurable": {"thread_id": thread_id}})
    return state.values.get('messages', [])

# -------------------
# Session State Init
# -------------------
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = retrive_unique_thread()

add_thread(st.session_state['thread_id'])

# -------------------
# Sidebar UI
# -------------------
st.sidebar.title("Start Conversation")

if st.sidebar.button("New Chat"):
    reset_chat()
    save_thread_metadata(str(st.session_state['thread_id']), {"thread_name": "New Chat"})

st.sidebar.markdown("---")
st.sidebar.header("My Conversations")

for tid in st.session_state['chat_threads']:
    meta = thread_document_metadata(str(tid))
    # ✅ DB se thread_name milta hai — agar nahi mila to fallback short id
    label = meta.get("thread_name") or (str(tid)[:8] + "...")
    label = label[:30]
    if thread_has_document(str(tid)):
        label = "📄 " + label

    if st.sidebar.button(label, key=str(tid)):
        st.session_state['thread_id'] = tid
        messages = load_conversations(tid)
        temp_messages = []
        for msg in messages:
            role = 'user' if isinstance(msg, HumanMessage) else 'assistant'
            temp_messages.append({'role': role, 'content': msg.content})
        st.session_state['message_history'] = temp_messages

# -------------------
# Main Chat UI
# -------------------
st.title("🤖 AI Research Assistant")

for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.text(message["content"])

upload_container = st.container()
with upload_container:
    current_tid = str(st.session_state['thread_id'])
    uploaded_file = st.file_uploader("Attach a PDF", type=["pdf"])

    if uploaded_file is not None:
        st.success(f"✅ {uploaded_file.name} uploaded")
        result = ingest_pdf(uploaded_file.read(), current_tid, uploaded_file.name)
        if result["success"]:
            st.info(f"📄 {uploaded_file.name} ready! ({result['pages']} pages, {result['chunks']} chunks)")
        else:
            st.error(f"❌ Error: {result['error']}")
    else:
        st.warning("📄 No PDF attached (removed or not uploaded)")
        if thread_has_document(current_tid):
            remove_thread_document(current_tid)
            clear_thread_checkpoint(current_tid)
            st.info("🗑️ Document context cleared for this thread")

user_input = st.chat_input("Type here...")

if user_input:
    current_tid = str(st.session_state['thread_id'])

    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.text(user_input)

    if uploaded_file is not None:
        if not thread_has_document(current_tid):
            result = ingest_pdf(
                file_bytes=uploaded_file.read(),
                thread_id=current_tid,
                filename=uploaded_file.name
            )
    else:
        remove_thread_document(current_tid)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        streamed_text = ""

        stream = workflow.stream(
            {"messages": [HumanMessage(content=user_input)]},
            config={
                "configurable": {"thread_id": st.session_state['thread_id']},
                "metadata": {"thread_id": st.session_state['thread_id']},
                "run_name": "chat_turn"
            },
            stream_mode="messages"
        )

        for chunk, metadata in stream:
            content = getattr(chunk, "content", "")

            # Empty content skip
            if not content:
                continue

            # tool/document classifier output skip
            if content.strip().lower() in ["tool", "document"]:
                continue

            # tools node ka raw output skip
            if metadata.get("langgraph_node") == "tools":
                continue

            streamed_text += content
            placeholder.text(streamed_text)

        st.session_state["message_history"].append({
            "role": "assistant",
            "content": streamed_text
        })

    # ✅ Pehla user message ke baad thread ka naam DB mein save karo (persistent), phir rerun karo
    #    taake sidebar turant updated naam dikhaye
    if len(st.session_state["message_history"]) == 2:
        save_thread_metadata(current_tid, {"thread_name": user_input[:30]})
        st.rerun()