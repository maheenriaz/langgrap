import logging
import streamlit as st
from langgraph_database_backend import (
    workflow, retrive_unique_thread,
    ingest_pdf, thread_has_document, thread_document_metadata,
    save_thread_metadata, remove_thread_document,
    clear_thread_checkpoint,
    DEFAULT_USER_ID, get_all_user_memories, delete_user_memory,
    clear_all_user_memories, remember_fact_explicitly
)
from langchain_core.messages import HumanMessage
import uuid
import re   # file ke top pe agar already nahi hai to add karo

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
    state = workflow.get_state({"configurable": {"thread_id": thread_id, "user_id": DEFAULT_USER_ID}})
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
st.sidebar.title("LangGraph Chatbot")

if st.sidebar.button("New Chat"):
    reset_chat()
    save_thread_metadata(str(st.session_state['thread_id']), {"thread_name": "New Chat"})

st.sidebar.markdown("---")
st.sidebar.header("My Conversations")

for tid in st.session_state['chat_threads']:
    meta = thread_document_metadata(str(tid))
    # DB se thread_name milta hai - agar nahi mila to fallback short id
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
# Sidebar — Long-Term Memory viewer
# -------------------
st.sidebar.markdown("---")
with st.sidebar.expander("🧠 What I remember about you"):
    memories = get_all_user_memories(DEFAULT_USER_ID)

    if not memories:
        st.caption("Abhi koi memory saved nahi hai. Jaise hi tum apna naam, preference, ya koi durable fact share karogi, yahan dikhega.")
    else:
        for mem in memories:
            col1, col2 = st.columns([5, 1])
            with col1:
                st.markdown(f"- {mem['text']}")
            with col2:
                if st.button("🗑️", key=f"del_{mem['key']}"):
                    delete_user_memory(mem['key'], DEFAULT_USER_ID)
                    st.rerun()

    st.markdown("---")
    manual_memory = st.text_input("Manually remember something:", key="manual_memory_input")
    if st.button("Save memory", key="save_manual_memory"):
        if manual_memory.strip():
            remember_fact_explicitly(manual_memory.strip(), DEFAULT_USER_ID)
            st.rerun()

    if memories and st.button("Clear all memories", key="clear_all_memories"):
        clear_all_user_memories(DEFAULT_USER_ID)
        st.rerun()

# -------------------
# Main Chat UI
# -------------------
st.title("🤖 AI Research Assistant by Maheen")

for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.text(message["content"])

# upload_container = st.container()
# with upload_container:
#     current_tid = str(st.session_state['thread_id'])
#     uploaded_file = st.file_uploader("Attach a PDF", type=["pdf"])

#     if uploaded_file is not None:
#         # st.success(f"✅ {uploaded_file.name} uploaded")
#         result = ingest_pdf(uploaded_file.read(), current_tid, uploaded_file.name)
#         # if result["success"]:
#         #     st.info(f"📄 {uploaded_file.name} ready! ({result['pages']} pages, {result['chunks']} chunks)")
#         # else:
#         #     st.error(f"❌ Error: {result['error']}")
#     else:
#         # st.warning("📄 No PDF attached (removed or not uploaded)")
#         if thread_has_document(current_tid):
#             remove_thread_document(current_tid)
#             clear_thread_checkpoint(current_tid)
#             # st.info("🗑️ Document context cleared for this thread")

user_input = st.chat_input("Type here...")

if user_input:
    current_tid = str(st.session_state['thread_id'])

    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.text(user_input)

    # if uploaded_file is not None:
    #     if not thread_has_document(current_tid):
    #         result = ingest_pdf(
    #             file_bytes=uploaded_file.read(),
    #             thread_id=current_tid,
    #             filename=uploaded_file.name
    #         )
    # else:
    #     remove_thread_document(current_tid)kk

    with st.chat_message("assistant"):
        placeholder = st.empty()
        streamed_text = ""

        stream = workflow.stream(
            {"messages": [HumanMessage(content=user_input)]},
            config={
                # ✅ user_id zaroori hai — remember_node aur routing node isko use karte hain
                # long-term memory ko sahi user ke namespace mein read/write karne ke liye
                "configurable": {
                    "thread_id": st.session_state['thread_id'],
                    "user_id": DEFAULT_USER_ID
                },
                "metadata": {"thread_id": st.session_state['thread_id']},
                "run_name": "chat_turn"
            },
            stream_mode="messages"
        )
        # print(f"[STREAMING RESPONSE] {stream}")
        sources_cutoff = False

        for chunk, metadata in stream:
            content = getattr(chunk, "content", "")
            if not content:
                continue
            if content.strip().lower() in ["tool", "document","yes","no"]:
                continue
            if metadata.get("langgraph_node") == "tools":
                continue
            if metadata.get("langgraph_node") == "remember":
                continue

            if sources_cutoff:
                continue   # sources shuru hone ke baad aane wale tokens ignore karo

            streamed_text += content

            # ✅ "Sources:" / "References:" pattern check karo
            match = re.search(r'\s*\**\s*(Sources?|References?|Citations?)\s*\**\s*:', streamed_text, re.IGNORECASE)
            if match:
                streamed_text = streamed_text[:match.start()].rstrip()
                sources_cutoff = True

            placeholder.text(streamed_text)
            st.session_state["message_history"].append({
                    "role": "assistant",
                    "content": streamed_text
            })

    # Pehla user message ke baad thread ka naam DB mein save karo (persistent), phir rerun karo
    # taake sidebar turant updated naam aur naya memory (agar koi save hua ho) dikhaye
    if len(st.session_state["message_history"]) == 2:
        save_thread_metadata(current_tid, {"thread_name": user_input[:30]})
        st.rerun()