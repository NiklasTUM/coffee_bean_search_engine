import streamlit as st
from src.rag_chain.RAGChain import RAGChain

rag_chain = RAGChain()

def main():
    st.title("RAG-Based Question Answering")

    # Input field for the user's question
    question = st.text_input("Enter your question:")

    # Button to submit the question
    if st.button("Get Answer"):
        if question:
            with st.spinner("Retrieving context and generating answer..."):
                # Call the chain method to get the answer
                answer = rag_chain.chain(question)
                st.success("Answer generated!")
                st.write(answer)
        else:
            st.warning("Please enter a question before submitting.")

    # Button to update the index
    if st.button("Update Index"):
        with st.spinner("Updating index..."):
            # Call the update_index method to update the index
            rag_chain.update_index()
            st.success("Index updated successfully!")


if __name__ == "__main__":
    main()
