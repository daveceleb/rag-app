"""
Utility script to generate embeddings for existing datasets in SQL Server.
Run this if you have datasets already in the database.
"""

import argparse
import sys
from embedding_pipeline import process_dataset_for_embeddings
import os

def main():
    parser = argparse.ArgumentParser(description="Generate embeddings for existing datasets")
    parser.add_argument("table_name", help="SQL Server table name (e.g., dataset_20251111_200000)")
    parser.add_argument("--api-key", help="Gemini API key (or set GEMINI_API_KEY env var)")
    parser.add_argument("--index-path", default="vector_store.index", help="Path to save FAISS index")
    parser.add_argument("--mapping-path", default="vector_map.json", help="Path to save vector mapping")
    
    args = parser.parse_args()
    
    # Get API key from argument or environment variable
    api_key = args.api_key or os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        print("❌ Error: Gemini API key not provided")
        print("   Either pass --api-key or set GEMINI_API_KEY environment variable")
        sys.exit(1)
    
    print(f"\n🚀 Generating embeddings for table: {args.table_name}")
    print(f"📁 Index path: {args.index_path}")
    print(f"📁 Mapping path: {args.mapping_path}")
    print("-" * 60)
    
    # Process dataset
    success, message, stats = process_dataset_for_embeddings(
        args.table_name,
        api_key,
        args.index_path,
        args.mapping_path
    )
    
    if success:
        print(f"\n✅ {message}")
        print("\n📊 Statistics:")
        for key, value in stats.items():
            print(f"   {key}: {value}")
    else:
        print(f"\n❌ {message}")
        sys.exit(1)

if __name__ == "__main__":
    main()
