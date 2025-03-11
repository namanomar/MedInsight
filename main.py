import os
import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint

# Constants
DB_FAISS_PATH = "vectorstore/db_faiss"
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
HF_TOKEN = os.environ.get("HUGGING_FACE_TOKEN")

st.set_page_config(page_title="Ask Chatbot!", page_icon="🤖", layout="wide")

# Function to load FAISS Vector Store
@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

# Custom Prompt
CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer the user's question.
If you don’t know the answer, just say that you don’t know. Don't try to make up an answer.
Only provide information from the given context.

Context: {context}
Question: {question}

Start the answer directly. No small talk, please.
"""

def set_custom_prompt(custom_prompt_template):
    return PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])

def load_llm(huggingface_repo_id, HF_TOKEN):
    return HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        model_kwargs={"token": HF_TOKEN, "max_length": "512"}
    )

def main():
    # Sidebar
    with st.sidebar:
        st.title("🤖 Chatbot Settings")
        st.markdown(""" 
        Customize your chatbot experience here. 
        - Uses **Mistral-7B** for responses
        - Retrieves top 3 most relevant sources
        """)
        st.info("Model: Mistral-7B | Retrieval: FAISS")
        
    # Chat UI
    st.title("💬 Ask Medical Chatbot!")
    st.write("Your AI-powered assistant. Type your question below:")
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])
    
    prompt = st.chat_input("Type your question here...")
    
    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})
        
        try:
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error("❌ Failed to load the vector store")
                return
            
            qa_chain = RetrievalQA.from_chain_type(
                llm=load_llm(huggingface_repo_id=HUGGINGFACE_REPO_ID, HF_TOKEN=HF_TOKEN),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                return_source_documents=True,
                chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )
            
            response = qa_chain.invoke({'query': prompt})
            result = response["result"]
            sources = response.get("source_documents", [])
            
            formatted_sources = "\n\n**Sources:**\n" + "\n".join([f"- {doc.metadata.get('source', 'Unknown')}" for doc in sources]) if sources else ""
            
            with st.chat_message('assistant'):
                st.markdown(f"**🤖 Chatbot:** {result}{formatted_sources}")
            
            st.session_state.messages.append({'role': 'assistant', 'content': f"**🤖 Chatbot:** {result}{formatted_sources}"})
            
        except Exception as e:
            st.error(f"⚠️ Error: {str(e)}")

if __name__ == "__main__":
    main()