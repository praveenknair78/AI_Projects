import RAG_engine
import streamlit as st

if __name__ == "__main__":

#    print(rag_bot.rag_query("What is encoder?", chat_history=None))
# --- Streamlit UI Code ---

    st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")

    st.title("🤖 RAG Chatbot")
    st.caption("Ask me anything about the documents I've been trained on.")

    # Initialize chat history in a Streamlit session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("What is your question?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Call your RAG core logic function here
                response = RAG_engine.rag_query(prompt)
                st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
