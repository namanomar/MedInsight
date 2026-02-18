
import streamlit as st
from langchain.chains import RetrievalQA

from llm.ollama_client import get_ollama_llm
from memory.vector_store import get_vector_store
from prompts.rag_prompt import get_rag_prompt, validate_prompt
from utils.helpers import format_sources, check_ollama_connection
import config

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="MedInsight",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading knowledge base...")
def get_vectorstore():
    """Load vector store"""
    vector_store = get_vector_store()
    return vector_store.load()


@st.cache_resource(show_spinner="Loading language model...")
def get_llm():
    """Load Ollama LLM"""
    if not check_ollama_connection(config.OLLAMA_BASE_URL):
        st.error(
            f"Cannot connect to Ollama at {config.OLLAMA_BASE_URL}. "
            f"Please ensure Ollama is running and pull the model: "
            f"`ollama pull {config.LLM_MODEL_NAME}`"
        )
        st.stop()
    return get_ollama_llm()


def get_qa_chain(vectorstore, llm) -> RetrievalQA:
    """Build QA chain"""
    prompt = get_rag_prompt(use_examples=config.USE_PROMPT_EXAMPLES)
    retriever = vectorstore.as_retriever(search_kwargs={"k": config.RETRIEVER_TOP_K})
    
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def render_sidebar() -> None:
    """Render sidebar with configuration info"""
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/caduceus.png", width=64)
        st.title("MedInsight")
        st.caption("AI-powered medical knowledge assistant")
        st.divider()
        
        st.markdown("**Model**")
        st.code(config.LLM_MODEL_NAME, language=None)
        st.caption(f"via Ollama ({config.OLLAMA_BASE_URL})")
        
        st.markdown("**Embeddings**")
        st.code(config.EMBEDDING_MODEL, language=None)
        
        st.markdown("**Retrieval**")
        st.markdown(f"Top-{config.RETRIEVER_TOP_K} most relevant documents")
        
        st.markdown("**Prompt Settings**")
        st.markdown(f"Examples: {'Enabled' if config.USE_PROMPT_EXAMPLES else 'Disabled'}")
        
        st.divider()
        
        if st.button("Clear chat history", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        
        st.caption(
            "Answers are grounded in the loaded medical knowledge base. "
            "Always consult a qualified healthcare professional."
        )


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main() -> None:
    """Main application"""
    render_sidebar()

    st.title("🩺 MedInsight")
    st.write("Ask medical questions and get answers grounded in your knowledge base.")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Replay history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # New user input
    user_input = st.chat_input("Ask a medical question...")
    if not user_input:
        return

    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base..."):
            try:
                # Validate input
                if len(user_input) > 1000:
                    st.error("Query is too long. Please keep it under 1000 characters.")
                    return
                
                vectorstore = get_vectorstore()
                llm = get_llm()
                qa_chain = get_qa_chain(vectorstore, llm)
                
                response = qa_chain.invoke({"query": user_input})

                answer = response["result"]
                sources = format_sources(response.get("source_documents", []))
                full_reply = f"{answer}{sources}"

                st.markdown(full_reply)
                st.session_state.messages.append({"role": "assistant", "content": full_reply})

            except Exception as exc:
                error_msg = f"An error occurred: {exc}"
                st.error(error_msg)
                st.exception(exc)


if __name__ == "__main__":
    main()
