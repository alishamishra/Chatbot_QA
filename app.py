import streamlit as st
import random
import time
from chatbot_core import create_retriever, create_graph
import tempfile
import os



#compile
graph = create_graph()

# 1. Initialize the session state flag
if "processed" not in st.session_state:
    st.session_state.processed = False


st.title("PDF QA chatbot")


#st.header("Upload your file")
uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file is not None and not st.session_state.processed:
    # 1. Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        # 2. Write the uploaded bytes to the temp file
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    # 3. Use the path in your function
    #st.write(f"The temporary file path is: {tmp_file_path}")
    retriever = create_retriever(tmp_file_path)
    st.success("PDF loaded, QA system is ready!! ")
    #Set the flag to True so it won't run again on the next rerun
    st.session_state.processed = True



# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display existing chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("How can I help?"):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Invoke LangGraph
    with st.chat_message("assistant"):
        inputs = {"messages": [("user", prompt)]}
        # Use .invoke() for simple responses or .stream() for real-time updates
        response = graph.invoke(inputs)
        
        # Extract the last message from the graph state
        final_msg = response["messages"][-1].content
        st.markdown(final_msg)
        st.session_state.messages.append({"role": "assistant", "content": final_msg})



    

