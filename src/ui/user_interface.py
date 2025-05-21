import streamlit as st
from src.rag_chain.RAGChain import RAGChain

# Initialize RAGChain in session state
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = RAGChain()

def main():
    st.sidebar.title("☕ AromaAtlas")
    st.sidebar.markdown("Find the perfect coffee, one flavor note at a time.")

    # Sidebar flavor sliders
    st.sidebar.header("Flavor Profile Filters")
    sweetness = st.sidebar.slider("Sweetness", 0, 10, 5)
    bitterness = st.sidebar.slider("Bitterness", 0, 10, 5)
    acidity = st.sidebar.slider("Acidity", 0, 10, 5)
    body = st.sidebar.slider("Body (light to full)", 0, 10, 5)

    st.title("Coffee Bean Search")

    # Input field for search
    search_query = st.text_input("Describe your ideal coffee:",
                                 placeholder="e.g., fruity, chocolatey, low-acidity")


    # Submit button
    if st.button("Search Beans"):
        if search_query.strip():
            # Combine input + sliders
            flavor_profile = (
                f"Sweetness: {sweetness}/10, "
                f"Bitterness: {bitterness}/10, "
                f"Acidity: {acidity}/10, "
                f"Body: {body}/10"
            )
            full_query = f"{search_query.strip()}. {flavor_profile}"

            with st.spinner("Searching coffee beans..."):
                answer = st.session_state.rag_chain.chain(full_query)
                st.success("Here are your matches:")
                st.text(answer)
        else:
            st.warning("Please enter some flavor notes to begin searching.")

    # Button to update the index
    if st.button("Update Index"):
        with st.spinner("Updating index..."):
            # Call the update_index method to update the index
            st.session_state.rag_chain.update_index()
            st.success("Index updated successfully!")

if __name__ == "__main__":
    main()
