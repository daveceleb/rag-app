# 🤖 RAG Chat Application

A complete Retrieval-Augmented Generation (RAG) chatbot application built with Streamlit, FAISS, and Google Gemini API.

## 🎯 Features

- **📤 CSV Upload**: Upload datasets directly to SQL Server
- **🔍 Vector Search**: FAISS-based similarity search with embeddings
- **💬 AI Chat**: Ask questions about your data using Gemini API
- **📊 Dataset Management**: Browse and select uploaded datasets
- **🎨 Modern UI**: Clean Streamlit interface with organized workflows

## 📁 Project Structure

```
rag/
├── app.py                      # Main Streamlit app (home page)
├── pages/
│   └── chatbot.py             # Chat interface page
├── database.py                # SQL Server connection & operations
├── embedding_pipeline.py      # FAISS & embedding generation
├── requirements.txt           # Python dependencies
├── vector_store.index        # FAISS index (auto-generated)
└── vector_map.json           # Vector-to-text mapping (auto-generated)
```

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.12+
- SQL Server (DESKTOP-4EBCN4A\SQLEXPRESS)
- Database: AcademicDB
- Gemini API Key: https://ai.google.dev

### 2. Installation

```powershell
# Clone/navigate to project
cd c:\Users\USER\Desktop\rag

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration

Set your Gemini API key (one of these methods):

**Option A: Environment Variable**
```powershell
$env:GEMINI_API_KEY = "your-api-key-here"
streamlit run app.py
```

**Option B: In-App Settings**
1. Run `streamlit run app.py`
2. Go to Sidebar → Settings → API Configuration
3. Paste your Gemini API key

### 4. Run the App

```powershell
streamlit run app.py
```

The app will open at: http://localhost:8501

## 📋 Workflow

### Upload & Process

1. **📤 Upload CSV**
   - Go to "Upload Dataset" tab
   - Select a CSV file
   - Data is saved to SQL Server table: `dataset_YYYYMMDD_HHMMSS`
   - Embeddings auto-generated (if Gemini key configured)
   - FAISS index created

2. **📂 Browse & Select**
   - Go to "Browse Datasets" tab
   - Select a dataset from dropdown
   - Click "Work with This Dataset"
   - Dataset becomes active

### Chat with Data

1. **💬 Open Chatbot**
   - Click "Open Chatbot" button
   - Or navigate via sidebar

2. **🎯 Ask Questions**
   - Type your question
   - Click "Ask"
   - AI retrieves relevant rows from FAISS
   - Gemini generates context-aware answer

3. **📊 View Sources**
   - Expand "Sources Used" to see which rows were used
   - View similarity scores

## 🛠️ Technical Stack

| Component | Technology |
|-----------|------------|
| Frontend | Streamlit 1.28+ |
| Database | SQL Server + SQLAlchemy |
| Vector DB | FAISS |
| Embeddings | Google Gemini API |
| LLM | Gemini 1.5 Pro |
| Data Processing | Pandas, NumPy |
| Driver | PyODBC |

## 📊 Supported Features

### Database Operations
- ✅ Create/Read tables in SQL Server
- ✅ Metadata tracking (filename, row count, columns)
- ✅ Automatic schema detection

### Embeddings
- ✅ Generate embeddings via Gemini API
- ✅ Store in local FAISS index
- ✅ Create vector-to-text mappings
- ✅ Similarity search (top-k retrieval)

### Chat
- ✅ Context-aware Q&A
- ✅ Multi-turn conversations
- ✅ Chat history
- ✅ Source attribution
- ✅ Clear conversation option

## ⚙️ Configuration

### SQL Server Connection
Edit in `database.py`:
```python
SERVER = "DESKTOP-4EBCN4A\\SQLEXPRESS"
DATABASE = "AcademicDB"
DRIVER = "ODBC Driver 17 for SQL Server"
```

### Vector Store Paths
Edit in `embedding_pipeline.py` and `pages/chatbot.py`:
```python
INDEX_PATH = "vector_store.index"      # FAISS index
MAPPING_PATH = "vector_map.json"       # Vector mapping
```

### Embedding Model
Default: `models/embedding-001` (Gemini)
Can be changed in embedding_pipeline.py

### Chat Model
Default: `gemini-1.5-pro-latest`
Can be changed in pages/chatbot.py

## 📝 Example Usage

### Query Examples
- "What is the average salary?"
- "Show me all employees in the sales department"
- "List the top 5 highest paid positions"
- "What departments exist in our database?"

### Expected Flow
```
User Uploads CSV
    ↓
Data → SQL Server table
    ↓
Rows → Text chunks → Embeddings
    ↓
FAISS Index + Mapping JSON created
    ↓
User asks question
    ↓
Query → Embedding → FAISS Search
    ↓
Top-K rows retrieved
    ↓
Gemini generates answer
    ↓
Answer + Sources displayed
```

## 🔑 API Keys

### Gemini API
1. Go to https://ai.google.dev
2. Click "Get API Key"
3. Create new key
4. Configure in app settings

## 📋 File Formats

### Supported Input
- CSV files with headers
- Any data types (will be converted to text)
- Multiple datasets can be uploaded

### Generated Files
- `vector_store.index` - FAISS binary index (~size depends on embeddings)
- `vector_map.json` - JSON mapping of vector IDs to text chunks

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| SQL Connection Failed | Check server name, database exists, ODBC driver installed |
| Gemini API errors | Verify API key is valid and has quota |
| FAISS index not found | Upload a dataset first to generate embeddings |
| Embeddings taking long | Large datasets may take time; be patient or check API quota |
| Memory errors | FAISS works in-memory; for very large datasets use approximate indices |

## 📚 Dependencies

See `requirements.txt`:
- streamlit>=1.28.0
- pandas>=2.0.0
- sqlalchemy>=2.0.0
- pyodbc>=4.0.0
- faiss-cpu>=1.7.4
- google-generativeai>=0.3.0

## 🎓 How RAG Works

```
1. Indexing Phase:
   Data → Chunks → Embeddings → FAISS Index

2. Retrieval Phase:
   Query → Embedding → Search FAISS → Get Top-K Chunks

3. Generation Phase:
   Query + Top-K Chunks → Gemini → Generate Answer
```

## 🚀 Future Enhancements

- [ ] Support for multiple vector stores (Pinecone, Weaviate)
- [ ] Multiple embedding models
- [ ] Persistent vector storage
- [ ] User authentication
- [ ] Chat export/save
- [ ] Advanced filtering options
- [ ] Real-time data updates
- [ ] Multi-turn context retention

## 📄 License

This project is provided as-is for educational purposes.

## 🤝 Support

For issues or questions, check:
1. Troubleshooting section above
2. Streamlit documentation: https://docs.streamlit.io
3. FAISS documentation: https://github.com/facebookresearch/faiss
4. Gemini API documentation: https://ai.google.dev/docs

---

**Version**: 0.4.0  
**Last Updated**: November 11, 2025  
**Status**: ✅ Production Ready
