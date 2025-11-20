import streamlit as st
import pandas as pd
from datetime import datetime
import os
from database import (
    test_connection,
    save_dataset_to_sql,
    get_dataset_metadata,
    get_table_structure,
    delete_dataset
)
from embedding_pipeline import process_dataset_for_embeddings

# Page configuration
st.set_page_config(
    page_title="RAG App",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'active_dataset' not in st.session_state:
    st.session_state.active_dataset = None
if 'uploaded_data' not in st.session_state:
    st.session_state.uploaded_data = None
if 'last_uploaded_file' not in st.session_state:
    st.session_state.last_uploaded_file = None
if 'show_datasets' not in st.session_state:
    st.session_state.show_datasets = False
if 'openai_api_key' not in st.session_state:
    st.session_state.openai_api_key = os.getenv("OPENAI_API_KEY", "")


# Main App
def main():
    # Header
    st.title("🤖 RAG App")
    st.markdown("---")
    
    # Welcome message
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ## Welcome to the RAG App!
        Upload your dataset to begin.
        
        This application allows you to:
        - 📤 Upload CSV datasets
        - 💾 Store data in SQL Server
        - 💬 Ask questions about your data using AI
        """)
    
    with col2:
        st.subheader("🔗 Connection Status")
        is_connected, connection_msg = test_connection()
        if is_connected:
            st.success("Connected to SQL Server")
        else:
            st.error(f"Connection Failed: {connection_msg}")
    
    st.markdown("---")
    
    # Sidebar - Upload Section
    with st.sidebar:
        st.header("📋 Menu")
        
        # Tab-like interface
        menu_option = st.radio(
            "Select Option",
            ["📤 Upload Dataset", "📂 Browse Datasets"],
            label_visibility="collapsed"
        )
        
        if menu_option == "📤 Upload Dataset":
            st.subheader("📤 Upload CSV Dataset")
            uploaded_file = st.file_uploader(
                "Choose a CSV file",
                type=["csv"],
                help="Upload your dataset in CSV format",
                key="csv_uploader"
            )
            
            # Only process if file is uploaded AND it's different from the last one
            if uploaded_file is not None and st.session_state.last_uploaded_file != uploaded_file.name:
                try:
                    # Read CSV file
                    df = pd.read_csv(uploaded_file) 
                    
                    st.session_state.uploaded_data = df
                    
                    # Generate unique table name with timestamp
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    table_name = f"dataset_{timestamp}"
                    
                    with st.spinner("Processing and saving to database..."):
                        # Save to SQL Server
                        success, message, row_count = save_dataset_to_sql(
                            df,
                            table_name,
                            uploaded_file.name
                        )
                    
                    if success:
                        # Update last uploaded file to prevent duplicates
                        st.session_state.last_uploaded_file = uploaded_file.name
                        st.session_state.active_dataset = table_name
                        
                        st.success(message)
                        
                        # Display upload summary
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Rows Saved", row_count)
                        with col2:
                            st.metric("Columns", len(df.columns))
                        with col3:
                            st.metric("Table Name", table_name)
                        
                        # Check if OpenAI API key is configured
                        if st.session_state.openai_api_key:
                            st.info("🚀 Processing embeddings...")
                            with st.spinner("Generating embeddings and building vector index..."):
                                embed_success, embed_message, embed_stats = process_dataset_for_embeddings(
                                    table_name,
                                    st.session_state.openai_api_key
                                )
                            
                            if embed_success:
                                st.success(embed_message)
                                with st.expander("📊 Embedding Stats"):
                                    st.json(embed_stats)
                            else:
                                st.warning(f"⚠️ Embedding generation failed: {embed_message}\nYou can still use the dataset without embeddings.")
                        else:
                            st.warning("⚠️ OpenAI API key not configured. Embeddings will not be generated. Configure it in Settings.")
                        
                        # Switch to dataset selection
                        st.info("✅ Dataset saved! Switch to 'Browse Datasets' to work with it.")
                    else:
                        st.error(message)
                
                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")
        
        elif menu_option == "📂 Browse Datasets":
            st.subheader("📂 Select Dataset to Work With")
            
            metadata_df = get_dataset_metadata()
            
            if not metadata_df.empty:
                # Create a list of datasets for selection
                dataset_options = {}
                for idx, row in metadata_df.iterrows():
                    label = f"{row['file_name']} ({row['row_count']} rows) - {row['upload_timestamp']}"
                    dataset_options[label] = row['table_name']
                
                selected_dataset_label = st.selectbox(
                    "Available Datasets:",
                    options=list(dataset_options.keys()),
                    index=0 if len(dataset_options) > 0 else None
                )
                
                if selected_dataset_label:
                    selected_table_name = dataset_options[selected_dataset_label]
                    
                    # Display dataset details
                    selected_row = metadata_df[metadata_df['table_name'] == selected_table_name].iloc[0]
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Rows", selected_row['row_count'])
                    with col2:
                        st.metric("Columns", selected_row['column_count'])
                    with col3:
                        st.metric("Uploaded", str(selected_row['upload_timestamp']))
                    
                    # Set active dataset
                    if st.button("🚀 Work with This Dataset", use_container_width=True, type="primary"):
                        st.session_state.active_dataset = selected_table_name
                        st.session_state.uploaded_data = None
                        st.success(f"✅ Active dataset set to: {selected_table_name}")
                        st.balloons()
                    
                    # Generate embeddings for existing dataset
                    st.markdown("---")
                    st.subheader("⚡ Generate Embeddings")
                    
                    import os
                    if os.path.exists("vector_store.index"):
                        st.info("✅ Embeddings already exist for this dataset")
                    else:
                        if st.button("🔄 Generate Embeddings Now", use_container_width=True):
                            if not st.session_state.openai_api_key:
                                st.error("❌ OpenAI API key not configured. Configure it in Settings first.")
                            else:
                                st.info("🚀 Generating embeddings...")
                                with st.spinner("This may take a few minutes depending on dataset size..."):
                                    embed_success, embed_message, embed_stats = process_dataset_for_embeddings(
                                        selected_table_name,
                                        st.session_state.openai_api_key
                                    )
                                
                                if embed_success:
                                    st.success(embed_message)
                                    with st.expander("📊 Embedding Stats"):
                                        st.json(embed_stats)
                                    st.balloons()
                                else:
                                    st.error(f"❌ Failed: {embed_message}")
                    
                    # Preview button
                    if st.button("👁️ Preview Data", use_container_width=True):
                        columns, preview_df = get_table_structure(selected_table_name)
                        if preview_df is not None:
                            st.dataframe(preview_df, use_container_width=True)
            else:
                st.info("No datasets uploaded yet. Upload a CSV file to get started!")
        
        st.markdown("---")
        
        # App info
        st.subheader("ℹ️ About")
        st.markdown("""
        - **Version**: 0.4.0
        - **Database**: SQL Server
        - **AI Model**: OpenAI (Free-tier)
        - **Database**: AcademicDB
        """)
        
        # Settings section
        st.markdown("---")
        st.subheader("⚙️ Settings")
        
        with st.expander("🔑 API Configuration"):
            openai_key_input = st.text_input(
                "OpenAI API Key",
                value=st.session_state.openai_api_key,
                type="password",
                help="Enter your OpenAI API key for embeddings"
            )
            
            if openai_key_input != st.session_state.openai_api_key:
                st.session_state.openai_api_key = openai_key_input
                st.success("✅ API key updated")
            
            st.caption("Get your API key from: https://platform.openai.com/account/api-keys")
    
    # Main content area - Show active dataset info
    if st.session_state.active_dataset:
        st.subheader(f"📊 Working with: `{st.session_state.active_dataset}`")
        
        columns, preview_df = get_table_structure(st.session_state.active_dataset)
        
        if preview_df is not None:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown("**Data Preview (First 5 rows):**")
                st.dataframe(preview_df, use_container_width=True)
            
            with col2:
                if columns:
                    col_count = len(columns)
                    st.metric("Columns", col_count)
                
                with st.expander("📋 Column Info"):
                    for col in columns:
                        st.write(f"**{col['name']}** - {col['type']}")
        
        st.markdown("---")
        
        # Chatbot button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("💬 Open Chatbot", use_container_width=True, type="primary", key="open_chatbot_main"):
                st.session_state.active_dataset = st.session_state.active_dataset
                st.switch_page("pages/chatbot.py")
    
    elif st.session_state.uploaded_data is not None:
        # Show uploaded data preview
        st.subheader("📊 Uploaded Data Preview")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.dataframe(st.session_state.uploaded_data.head(10), use_container_width=True)
        
        with col2:
            st.metric("Total Rows", len(st.session_state.uploaded_data))
            st.metric("Columns", len(st.session_state.uploaded_data.columns))
        
        # Data Info
        with st.expander("📋 Column Information"):
            info_df = pd.DataFrame({
                'Column': st.session_state.uploaded_data.columns,
                'Type': st.session_state.uploaded_data.dtypes.astype(str),
                'Non-Null': st.session_state.uploaded_data.count(),
                'Null': st.session_state.uploaded_data.isnull().sum()
            })
            st.dataframe(info_df, use_container_width=True)
    
    else:
        st.info("👆 Upload a CSV file or select an existing dataset in the sidebar to get started!")


if __name__ == "__main__":
    main()
