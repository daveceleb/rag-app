"""
Embedding pipeline for generating embeddings and building FAISS vector store.
"""

import json
import logging
import numpy as np
import pandas as pd
import faiss
import google.generativeai as genai
from database import get_engine, get_table_structure
from sqlalchemy import text
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Gemini API
def configure_gemini(api_key):
    """Configure Gemini API with the provided key."""
    try:
        genai.configure(api_key=api_key)
        logger.info("Gemini API configured successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to configure Gemini API: {str(e)}")
        return False


def get_embedding(text, model="models/embedding-001"):
    """
    Generate embedding for a given text using Gemini Embeddings API.
    
    Args:
        text (str): Text to embed
        model (str): Embedding model to use
        
    Returns:
        list: Embedding vector, or None if error
    """
    try:
        response = genai.embed_content(
            model=model,
            content=text
        )
        return response['embedding']
    except Exception as e:
        logger.error(f"Error generating embedding: {str(e)}")
        return None


def create_text_chunks(df):
    """
    Convert DataFrame rows into text chunks.
    Each row becomes a single text chunk by joining all column values.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        
    Returns:
        list: List of text chunks
    """
    try:
        text_chunks = []
        for idx, row in df.iterrows():
            # Join all column values into a single string
            chunk = ' '.join([f"{col}: {str(val)}" for col, val in row.items()])
            text_chunks.append(chunk)
        
        logger.info(f"Created {len(text_chunks)} text chunks from dataset")
        return text_chunks
    except Exception as e:
        logger.error(f"Error creating text chunks: {str(e)}")
        return []


def build_faiss_index(embeddings, index_path="vector_store.index"):
    """
    Build FAISS index from embeddings and save it locally.
    
    Args:
        embeddings (list): List of embedding vectors
        index_path (str): Path to save FAISS index
        
    Returns:
        tuple: (success: bool, index_path: str, message: str)
    """
    try:
        if not embeddings or len(embeddings) == 0:
            return False, None, "No embeddings to build index"
        
        # Convert to numpy array
        embeddings_array = np.array(embeddings).astype('float32')
        
        # Get embedding dimension
        dim = embeddings_array.shape[1]
        
        # Create FAISS index
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings_array)
        
        # Save index
        faiss.write_index(index, index_path)
        
        message = f"✅ FAISS index created with {len(embeddings)} embeddings (dim: {dim})"
        logger.info(message)
        return True, index_path, message
    
    except Exception as e:
        error_msg = f"Error building FAISS index: {str(e)}"
        logger.error(error_msg)
        return False, None, error_msg


def save_vector_mapping(text_chunks, table_name, mapping_path="vector_map.json"):
    """
    Save mapping between vector IDs and original text chunks.
    
    Args:
        text_chunks (list): List of text chunks
        table_name (str): Name of the dataset table
        mapping_path (str): Path to save mapping JSON
        
    Returns:
        tuple: (success: bool, mapping_path: str, message: str)
    """
    try:
        mapping = {
            "table_name": table_name,
            "total_vectors": len(text_chunks),
            "vectors": {
                str(idx): {
                    "id": idx,
                    "text": chunk
                }
                for idx, chunk in enumerate(text_chunks)
            }
        }
        
        with open(mapping_path, 'w', encoding='utf-8') as f:
            json.dump(mapping, f, indent=2, ensure_ascii=False)
        
        message = f"✅ Vector mapping saved with {len(text_chunks)} entries"
        logger.info(message)
        return True, mapping_path, message
    
    except Exception as e:
        error_msg = f"Error saving vector mapping: {str(e)}"
        logger.error(error_msg)
        return False, None, error_msg


def load_vector_mapping(mapping_path="vector_map.json"):
    """
    Load vector mapping from JSON file.
    
    Args:
        mapping_path (str): Path to mapping JSON
        
    Returns:
        dict: Mapping dictionary, or None if error
    """
    try:
        if not os.path.exists(mapping_path):
            logger.warning(f"Mapping file not found: {mapping_path}")
            return None
        
        with open(mapping_path, 'r', encoding='utf-8') as f:
            mapping = json.load(f)
        
        logger.info(f"Loaded mapping with {mapping.get('total_vectors', 0)} vectors")
        return mapping
    
    except Exception as e:
        logger.error(f"Error loading vector mapping: {str(e)}")
        return None


def process_dataset_for_embeddings(table_name, gemini_api_key, 
                                  index_path="vector_store.index",
                                  mapping_path="vector_map.json"):
    """
    Complete pipeline: fetch dataset, generate embeddings, build FAISS index.
    
    Args:
        table_name (str): Name of the dataset table in SQL Server
        gemini_api_key (str): Gemini API key
        index_path (str): Path to save FAISS index
        mapping_path (str): Path to save vector mapping
        
    Returns:
        tuple: (success: bool, message: str, stats: dict)
    """
    try:
        # Configure Gemini API
        if not configure_gemini(gemini_api_key):
            return False, "Failed to configure Gemini API", {}
        
        # Fetch dataset from SQL Server
        logger.info(f"Fetching dataset from table: {table_name}")
        engine = get_engine()
        query = f"SELECT * FROM [{table_name}]"
        df = pd.read_sql(query, con=engine)
        
        if df.empty:
            return False, "Dataset is empty", {}
        
        logger.info(f"Fetched {len(df)} rows from {table_name}")
        
        # Create text chunks
        text_chunks = create_text_chunks(df)
        if not text_chunks:
            return False, "Failed to create text chunks", {}
        
        # Generate embeddings
        logger.info("Generating embeddings for all chunks...")
        embeddings = []
        failed_count = 0
        
        for idx, chunk in enumerate(text_chunks):
            embedding = get_embedding(chunk)
            if embedding is not None:
                embeddings.append(embedding)
            else:
                failed_count += 1
            
            # Log progress every 10 chunks
            if (idx + 1) % 10 == 0:
                logger.info(f"Generated {idx + 1}/{len(text_chunks)} embeddings")
        
        if not embeddings:
            return False, "Failed to generate embeddings", {}
        
        logger.info(f"Successfully generated {len(embeddings)} embeddings (failed: {failed_count})")
        
        # Build FAISS index
        success, saved_index_path, index_msg = build_faiss_index(embeddings, index_path)
        if not success:
            return False, index_msg, {}
        
        # Save vector mapping
        success, saved_mapping_path, mapping_msg = save_vector_mapping(text_chunks, table_name, mapping_path)
        if not success:
            return False, mapping_msg, {}
        
        # Prepare statistics
        stats = {
            "table_name": table_name,
            "total_rows": len(df),
            "total_chunks": len(text_chunks),
            "total_embeddings": len(embeddings),
            "embedding_dimension": len(embeddings[0]) if embeddings else 0,
            "failed_embeddings": failed_count,
            "index_path": saved_index_path,
            "mapping_path": saved_mapping_path
        }
        
        final_message = f"✅ Dataset indexed successfully!\n- Rows processed: {len(df)}\n- Embeddings created: {len(embeddings)}\n- Embedding dimension: {len(embeddings[0])}"
        
        logger.info(f"Pipeline completed successfully: {final_message}")
        return True, final_message, stats
    
    except Exception as e:
        error_msg = f"Error in embedding pipeline: {str(e)}"
        logger.error(error_msg)
        return False, error_msg, {}


def search_similar(query_text, gemini_api_key, index_path="vector_store.index", 
                  mapping_path="vector_map.json", top_k=5):
    """
    Search for similar content in the FAISS index.
    
    Args:
        query_text (str): Query text
        gemini_api_key (str): Gemini API key
        index_path (str): Path to FAISS index
        mapping_path (str): Path to vector mapping
        top_k (int): Number of top results to return
        
    Returns:
        tuple: (success: bool, results: list, message: str)
    """
    try:
        # Configure Gemini API
        configure_gemini(gemini_api_key)
        
        # Generate query embedding
        query_embedding = get_embedding(query_text)
        if query_embedding is None:
            return False, [], "Failed to generate query embedding"
        
        # Load FAISS index
        if not os.path.exists(index_path):
            return False, [], f"Index file not found: {index_path}"
        
        index = faiss.read_index(index_path)
        
        # Load vector mapping
        mapping = load_vector_mapping(mapping_path)
        if mapping is None:
            return False, [], f"Mapping file not found: {mapping_path}"
        
        # Search
        query_array = np.array([query_embedding]).astype('float32')
        distances, indices = index.search(query_array, min(top_k, mapping['total_vectors']))
        
        # Build results
        results = []
        for rank, (idx, distance) in enumerate(zip(indices[0], distances[0])):
            if str(idx) in mapping['vectors']:
                result = {
                    "rank": rank + 1,
                    "id": int(idx),
                    "distance": float(distance),
                    "similarity": float(1 / (1 + distance)),  # Convert distance to similarity
                    "text": mapping['vectors'][str(idx)]['text']
                }
                results.append(result)
        
        logger.info(f"Found {len(results)} similar results for query")
        return True, results, "Search successful"
    
    except Exception as e:
        error_msg = f"Error searching FAISS index: {str(e)}"
        logger.error(error_msg)
        return False, [], error_msg
