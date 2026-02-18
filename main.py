"""
MedInsight - Streamlit Web Application
RAG-powered medical chatbot using FAISS + Mistral-7B
"""
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

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
def get_vectorstore() -> FAISS:
    embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
    return FAISS.load_local(
        config.DB_FAISS_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
    )


@st.cache_resource(show_spinner="Loading language model...")
def get_llm() -> HuggingFaceEndpoint:
    return HuggingFaceEndpoint(
        repo_id=config.LLM_REPO_ID,
        temperature=config.LLM_TEMPERATURE,
        model_kwargs={
            "token": config.HF_TOKEN,
            "max_new_tokens": config.LLM_MAX_NEW_TOKENS,
        },
    )


def get_qa_chain(vectorstore: FAISS, llm: HuggingFaceEndpoint) -> RetrievalQA:
    prompt = PromptTemplate(
        template=config.PROMPT_TEMPLATE,
        input_variables=["context", "question"],
    )
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": config.RETRIEVER_TOP_K}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def render_sidebar() -> None:
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/caduceus.png", width=64)
        st.title("MedInsight")
        st.caption("AI-powered medical knowledge assistant")
        st.divider()
        st.markdown("**Model**")
        st.code(config.LLM_REPO_ID, language=None)
        st.markdown("**Embeddings**")
        st.code(config.EMBEDDING_MODEL, language=None)
        st.markdown("**Retrieval**")
        st.markdown(f"Top-{config.RETRIEVER_TOP_K} most relevant documents")
        st.divider()
        if st.button("Clear chat history", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        st.caption("Answers are grounded in the loaded medical knowledge base. Always consult a qualified healthcare professional.")


def render_sources(source_docs: list) -> str:
    if not source_docs:
        return ""
    seen = set()
    lines = []
    for doc in source_docs:
        src = doc.metadata.get("source", "Unknown source")
        page = doc.metadata.get("page", "")
        label = f"{src}" + (f" — p.{page + 1}" if page != "" else "")
        if label not in seen:
            seen.add(label)
            lines.append(f"- {label}")
    return "\n\n**Sources:**\n" + "\n".join(lines)


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main() -> None:
    render_sidebar()

    st.title("🩺 MedInsight")
    st.write("Ask medical questions and get answers grounded in your knowledge base.")

    # Initialise chat history
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
                vectorstore = get_vectorstore()
                llm = get_llm()
                qa_chain = get_qa_chain(vectorstore, llm)
                response = qa_chain.invoke({"query": user_input})

                answer = response["result"]
                sources = render_sources(response.get("source_documents", []))
                full_reply = f"{answer}{sources}"

                st.markdown(full_reply)
                st.session_state.messages.append({"role": "assistant", "content": full_reply})

            except Exception as exc:
                error_msg = f"An error occurred: {exc}"
                st.error(error_msg)


if __name__ == "__main__":
    main()
