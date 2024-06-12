import streamlit as st
from chatbot import OpenAIChatBot, SYSTEM_MESSAGE_PROMPT
from langchain_core.messages import AIMessage, HumanMessage

# from dotenv import load_dotenv
# import os

# load_dotenv()
# uploaded_files = st.sidebar.file_uploader("Upload image", type=['png', 'jpg', 'pdf'], accept_multiple_files=True)

st.title("Repro-Tox Demo")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message.type):
        st.markdown(message.content)

st.session_state.bot = OpenAIChatBot()

if prompt := st.chat_input("Text here..."):
    with st.chat_message("User"):
        st.markdown(prompt)

    with st.chat_message("AI"):
        message_placeholder = st.empty()
        full_response = st.session_state.bot.query(
            prompt,
            chat_history=st.session_state.messages
        )
        message_placeholder.markdown(full_response)

    st.session_state.messages.append(AIMessage(content=full_response))