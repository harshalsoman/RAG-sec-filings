import streamlit as st
from chat import run_chat

# Set page configuration first
st.set_page_config(page_title="Query-Response Chatbox")

if __name__ == "__main__":
    st.title("Query-Response Chatbox")
    st.write("Welcome to the real-time query-response chatbox!")
    st.write("Type your query below and get responses based on SEC filings.")
    st.write("---")
    run_chat()
