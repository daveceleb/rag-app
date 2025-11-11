"""
Chatbot page for RAG-based conversation about uploaded datasets.
"""

import streamlit as st
import json
import numpy as np
import faiss
import os
import logging
from embedding_pipeline import get_embedding, configure_gemini, load_vector_mapping
import google.generativeai as genai

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Chatbot",
    page_icon="💬",
    layout="wide"
)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'active_dataset' not in st.session_state:
    st.session_state.active_dataset = None
if 'gemini_api_key' not in st.session_state:
    st.session_state.gemini_api_key = os.getenv("GEMINI_API_KEY", "")


def load_faiss_index(index_path="vector_store.index"):
    """Load FAISS index from disk."""
    try:
        if not os.path.exists(index_path):
            return None, f"Index file not found: {index_path}"
        index = faiss.read_index(index_path)
        logger.info(f"FAISS index loaded successfully")
        return index, None
    except Exception as e:
        error_msg = f"Error loading FAISS index: {str(e)}"
        logger.error(error_msg)
        return None, error_msg


def retrieve_context(query, index, vector_map, gemini_api_key, top_k=5):
    """
    Retrieve context from FAISS index based on query similarity.
    
    Args:
        query (str): User query
        index: FAISS index
        vector_map (dict): Mapping of vector IDs to text
        gemini_api_key (str): Gemini API key for embeddings
        top_k (int): Number of top results to retrieve
        
    Returns:
        tuple: (success: bool, context_list: list, message: str)
    """
    try:
        if index is None:
            return False, [], "FAISS index not available"
        
        if vector_map is None or vector_map.get('total_vectors', 0) == 0:
            return False, [], "Vector mapping not available"
        
        # Configure Gemini for embedding
        configure_gemini(gemini_api_key)
        
        # Generate query embedding
        query_embedding = get_embedding(query)
        if query_embedding is None:
            return False, [], "Failed to generate query embedding"
        
        # Search FAISS index
        query_array = np.array([query_embedding]).astype('float32')
        distances, indices = index.search(query_array, min(top_k, vector_map['total_vectors']))
        
        # Extract context
        context_list = []
        for rank, (idx, distance) in enumerate(zip(indices[0], distances[0])):
            if str(idx) in vector_map['vectors']:
                context_item = {
                    "rank": rank + 1,
                    "id": int(idx),
                    "text": vector_map['vectors'][str(idx)]['text'],
                    "distance": float(distance),
                    "similarity": float(1 / (1 + distance))
                }
                context_list.append(context_item)
        
        logger.info(f"Retrieved {len(context_list)} context items for query")
        return True, context_list, "Context retrieved successfully"
    
    except Exception as e:
        error_msg = f"Error retrieving context: {str(e)}"
        logger.error(error_msg)
        return False, [], error_msg


def ask_gemini(query, context_list, gemini_api_key, model="gemini-1.5-pro-latest"):
    """
    Generate answer using Gemini based on query and context.
    
    Args:
        query (str): User query
        context_list (list): List of context items from retrieval
        gemini_api_key (str): Gemini API key
        model (str): Gemini model to use
        
    Returns:
        tuple: (success: bool, answer: str, message: str)
    """
    try:
        if not context_list:
            return False, "", "No context available to answer the question"
        
        # Configure Gemini
        configure_gemini(gemini_api_key)
        
        # Build context text
        context_text = "\n".join([
            f"[Source {item['rank']}] {item['text']}"
            for item in context_list
        ])
        
        # Build prompt
        prompt = f"""You are a helpful data assistant. Answer the following question using ONLY the provided context. 
If the answer is not in the context, say "I don't have enough information in the dataset to answer this question."

Context from dataset:
{context_text}

Question: {query}

Answer:"""
        
        # Generate response
        model_obj = genai.GenerativeModel(model)
        response = model_obj.generate_content(prompt)
        
        answer = response.text if response.text else "No response generated"
        logger.info(f"Generated answer successfully")
        return True, answer, "Answer generated successfully"
    
    except Exception as e:
        error_msg = f"Error generating answer: {str(e)}"
        logger.error(error_msg)
        return False, "", error_msg


def display_chat_history():
    """Display chat history in the UI."""
    if not st.session_state.chat_history:
        st.info("💭 Start the conversation by asking a question about your dataset!")
        return
    
    st.subheader("💬 Conversation History")
    
    for idx, chat in enumerate(st.session_state.chat_history):
        with st.container(border=True):
            col1, col2 = st.columns([0.1, 0.9])
            with col1:
                st.markdown("👤")
            with col2:
                st.markdown(f"**You:** {chat['query']}")
            
            col1, col2 = st.columns([0.1, 0.9])
            with col1:
                st.markdown("🤖")
            with col2:
                st.markdown(f"**Chatbot:** {chat['answer']}")
            
            # Show context if expanded
            with st.expander(f"📊 Sources (Chat #{idx + 1})"):
                if 'context' in chat:
                    for source in chat['context']:
                        st.caption(f"**Source {source['rank']}** (Similarity: {source['similarity']:.2%})")
                        st.write(source['text'])
                else:
                    st.info("No source context available")


def main():
    # Header
    st.title("💬 Chat with Your Dataset")
    st.write("Ask any question about your uploaded CSV dataset using AI-powered retrieval and generation.")
    
    # Check if dataset is active
    if not st.session_state.active_dataset:
        st.warning("⚠️ No active dataset selected. Please upload a CSV and select it from the main app first.")
        if st.button("← Go Back to Main App"):
            st.switch_page("app.py")
        return
    
    # Check if Gemini API key is configured
    if not st.session_state.gemini_api_key:
        st.error("❌ Gemini API key not configured. Please configure it in the main app settings.")
        return
    
    st.success(f"✅ Active Dataset: `{st.session_state.active_dataset}`")
    st.markdown("---")
    
    # Load FAISS index
    index, index_error = load_faiss_index()
    if index_error:
        st.warning(f"⚠️ {index_error}")
        st.info("Make sure embeddings have been generated for your dataset.")
        return
    
    # Load vector mapping
    vector_map = load_vector_mapping()
    if vector_map is None:
        st.warning("⚠️ Vector mapping not found. Make sure embeddings have been generated.")
        return
    
    st.info(f"📊 Loaded {vector_map.get('total_vectors', 0)} embeddings from dataset")
    st.markdown("---")
    
    # Chat interface
    st.subheader("🎯 Ask Your Question")
    
    # Create two columns for input and button
    col1, col2 = st.columns([4, 1])
    
    with col1:
        user_query = st.text_input(
            "Enter your question about the dataset:",
            placeholder="e.g., What is the average salary? or List all products...",
            key="chat_input"
        )
    
    with col2:
        ask_button = st.button("🚀 Ask", use_container_width=True, type="primary")
    
    # Process query
    if ask_button and user_query:
        with st.spinner("🔍 Retrieving context..."):
            success, context_list, retrieval_msg = retrieve_context(
                user_query,
                index,
                vector_map,
                st.session_state.gemini_api_key,
                top_k=5
            )
        
        if not success:
            st.error(f"❌ Retrieval failed: {retrieval_msg}")
            return
        
        st.success(f"✅ Retrieved {len(context_list)} relevant sources")
        
        with st.spinner("🤔 Generating answer..."):
            answer_success, answer, answer_msg = ask_gemini(
                user_query,
                context_list,
                st.session_state.gemini_api_key
            )
        
        if not answer_success:
            st.error(f"❌ Answer generation failed: {answer_msg}")
            return
        
        st.success(answer_msg)
        
        # Add to chat history
        st.session_state.chat_history.append({
            "query": user_query,
            "answer": answer,
            "context": context_list
        })
        
        # Display latest answer
        st.markdown("---")
        st.subheader("✨ Answer")
        st.markdown(answer)
        
        # Show sources
        with st.expander("📊 Sources Used", expanded=False):
            for source in context_list:
                st.caption(f"**Source {source['rank']}** - Similarity: {source['similarity']:.2%}")
                st.write(source['text'])
    
    elif ask_button and not user_query:
        st.warning("⚠️ Please enter a question first")
    
    st.markdown("---")
    
    # Display chat history
    if st.session_state.chat_history:
        display_chat_history()
    else:
        st.info("💭 Start the conversation by asking a question about your dataset!")
    
    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ Chat Settings")
        
        # Clear history button
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.chat_history = []
            st.success("Chat history cleared!")
            st.rerun()
        
        st.markdown("---")
        
        # Stats
        st.subheader("📈 Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Embeddings", vector_map.get('total_vectors', 0))
        with col2:
            st.metric("Chat Messages", len(st.session_state.chat_history) * 2)
        
        st.markdown("---")
        
        # Help section
        st.subheader("❓ Tips")
        st.markdown("""
        - Ask specific questions about your data
        - The AI uses the most relevant rows from your dataset
        - Questions are answered using context from your CSV
        - View sources to see which data was used
        """)
        
        # Back button
        st.markdown("---")
        if st.button("← Back to Main App", use_container_width=True):
            st.switch_page("app.py")


if __name__ == "__main__":
    main()
