import streamlit as st
import requests

st.set_page_config(page_title="UET AI Agent", page_icon="🎓")

st.title("🎓 UET Prospectus AI Agent")
st.caption("Powered by Gemma 3 & MCP")

# Session State for Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User Input
if prompt := st.chat_input("Ask about UET departments..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Fetch Response
    with st.chat_message("assistant"):
        with st.spinner("Agent is thinking..."):
            try:
                res = requests.post("http://localhost:8000/chat", json={"message": prompt})
                if res.status_code == 200:
                    answer = res.json()["response"]
                else:
                    answer = "Error: API returned an error."
            except:
                answer = "Error: Backend is not running. Please run 'python backend/main.py'."
            
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})