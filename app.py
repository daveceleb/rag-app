import streamlit as st
import os
import json
import pyodbc
import pandas as pd
import google.generativeai as genai

# ------------------------------------------------------
# Configuration
# ------------------------------------------------------
EMBED_MODEL = 'text-embedding-004'
TOP_N = 15

GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_API_KEY = "AIzaSyDVAAuONgaNH_KtMr-SNI4_Q7k3wH-RvEE"
genai.configure(api_key=GEMINI_API_KEY)

SQL_DRIVER = "ODBC Driver 17 for SQL Server"

# ------------------------------------------------------
# Streamlit Setup
# ------------------------------------------------------
st.set_page_config(page_title="Hybrid RAG Chatbot (CSV Version)", layout="wide")
st.title("📊 Hybrid RAG Chatbot (CSV + SQL Embeddings)")


# Session defaults
for key, default in {
    "history": [],
    "vector_db": [],
    "dataset": [],
    "current_file": None,
    "sql_connected": False,
    "sql_server": r"Obiba\SQLEXPRESS",
    "sql_database": "Datasets",
    "sql_username": "",
    "sql_password": "",
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ------------------------------------------------------
# Sidebar SQL Settings Panel
# ------------------------------------------------------
st.sidebar.header("🛠 SQL Server Settings")

st.session_state.sql_server = st.sidebar.text_input(
    "SQL Server Name", st.session_state.sql_server
)
st.session_state.sql_database = st.sidebar.text_input(
    "Database Name", st.session_state.sql_database
)
st.session_state.sql_username = st.sidebar.text_input(
    "SQL Username (optional)", st.session_state.sql_username
)
st.session_state.sql_password = st.sidebar.text_input(
    "SQL Password (optional)",
    st.session_state.sql_password,
    type="password"
)

# ------------------------------------------------------
# SQL CONNECTION HANDLER
# ------------------------------------------------------
def get_sql_connection():
    server = st.session_state.sql_server
    database = st.session_state.sql_database
    username = st.session_state.sql_username
    password = st.session_state.sql_password

    if username and password:
        conn_str = (
            f"DRIVER={{{SQL_DRIVER}}};"
            f"SERVER={server};"
            f"DATABASE={database};"
            f"UID={username};"
            f"PWD={password}"
        )
    else:
        conn_str = (
            f"DRIVER={{{SQL_DRIVER}}};"
            f"SERVER={server};"
            f"DATABASE={database};"
            f"Trusted_Connection=yes;"
        )
    return pyodbc.connect(conn_str)

# ------------------------------------------------------
# Validate SQL & Create Table
# ------------------------------------------------------
def create_table():
    try:
        conn = get_sql_connection()
        cursor = conn.cursor()
        cursor.execute("""
        IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='Embeddings' AND xtype='U')
        CREATE TABLE Embeddings (
            id INT IDENTITY(1,1) PRIMARY KEY,
            filename NVARCHAR(255),
            chunk NVARCHAR(MAX),
            embedding NVARCHAR(MAX)
        )
        """)
        conn.commit()
        cursor.close()
        conn.close()
        return True, "SQL connected & table ensured!"
    except Exception as e:
        return False, str(e)

if st.sidebar.button("Connect to SQL & Initialize"):
    ok, msg = create_table()
    if ok:
        st.session_state.sql_connected = True
        st.sidebar.success(msg)
    else:
        st.session_state.sql_connected = False
        st.sidebar.error(f"Connection failed: {msg}")

# ------------------------------------------------------
# Helper: List files from SQL
# ------------------------------------------------------
def list_existing_files():
    try:
        conn = get_sql_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT filename FROM Embeddings")
        files = [row[0] for row in cursor.fetchall()]
        cursor.close()
        conn.close()
        return files
    except Exception as e:
        st.sidebar.error(f"Error fetching files: {e}")
        return []

# ------------------------------------------------------
# Only allow file operations **after SQL is connected**
# ------------------------------------------------------
if st.session_state.sql_connected:

    st.sidebar.header("📂 File Management")

    existing_files = list_existing_files()
    selected_file = st.sidebar.selectbox("Select existing file", [""] + existing_files)

    if selected_file:
        st.session_state.current_file = selected_file
        conn = get_sql_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT chunk, embedding FROM Embeddings WHERE filename = ?", selected_file)
        rows = cursor.fetchall()
        cursor.close()
        conn.close()

        st.session_state.dataset = [row[0] for row in rows]
        st.session_state.vector_db = [(row[0], json.loads(row[1])) for row in rows]

        st.sidebar.success(f"Loaded '{selected_file}' from SQL Server!")

    # ------------------------------------------------------
    # Upload + Load / Compute Embeddings (CSV VERSION)
    # ------------------------------------------------------
    uploaded_file = st.sidebar.file_uploader("Upload a new CSV dataset", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        # Convert each row into a flattened text chunk
        st.session_state.dataset = [
            " | ".join([f"{col}: {str(row[col])}" for col in df.columns])
            for _, row in df.iterrows()
        ]

        st.session_state.current_file = uploaded_file.name
        st.session_state.vector_db = []
        st.sidebar.success(f"CSV '{uploaded_file.name}' loaded ({len(df)} rows)!")

    # ------------------------------------------------------
    # Embedding Generator
    # ------------------------------------------------------
    def embed_text(text):
        result = genai.embed_content(
            model=EMBED_MODEL,
            content=text,
            task_type="retrieval_document"
        )
        return result["embedding"]

    # ------------------------------------------------------
    # Save Embeddings
    # ------------------------------------------------------
    def save_embeddings_to_sql(filename, dataset, vector_db):
        conn = get_sql_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM Embeddings WHERE filename = ?", filename)

        for chunk, emb in vector_db:
            cursor.execute(
                "INSERT INTO Embeddings (filename, chunk, embedding) VALUES (?, ?, ?)",
                filename, chunk, json.dumps(emb)
            )

        conn.commit()
        cursor.close()
        conn.close()

    # ------------------------------------------------------
    # Load or Create Embeddings
    # ------------------------------------------------------
    def load_or_create_embeddings():
        if len(st.session_state.dataset) == 0:
            st.warning("No dataset loaded.")
            return

        filename = st.session_state.current_file

        if st.session_state.vector_db:
            st.success("Embeddings already loaded from SQL.")
            return

        temp_db = []
        progress = st.progress(0)
        total = len(st.session_state.dataset)

        for i, chunk in enumerate(st.session_state.dataset):
            try:
                emb = embed_text(chunk)
                temp_db.append((chunk, emb))
            except Exception as e:
                st.error(f"Embedding error on row {i+1}: {e}")

            progress.progress((i + 1) / total)

        st.session_state.vector_db = temp_db
        save_embeddings_to_sql(filename, st.session_state.dataset, temp_db)
        st.success("Embeddings computed & saved to SQL!")

    st.sidebar.header("Embedding Status")

    if st.sidebar.button("Load / Compute Embeddings"):
        load_or_create_embeddings()

    st.sidebar.write(f"📄 Rows in dataset: {len(st.session_state.dataset)}")
    st.sidebar.write(f"📦 Embeddings loaded: {len(st.session_state.vector_db)}")

# ------------------------------------------------------
# Cosine similarity
# ------------------------------------------------------
def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x**2 for x in a) ** 0.5
    norm_b = sum(y**2 for y in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0
    return dot / (norm_a * norm_b)

# ------------------------------------------------------
# Hybrid Retrieval
# ------------------------------------------------------
def retrieve(query, history, top_n=TOP_N):
    history_text = " ".join([h["user"] for h in history[-3:]])
    enhanced_query = history_text + " " + query if history else query
    query_embedding = embed_text(enhanced_query)
    scores = [(chunk, cosine_similarity(query_embedding, emb))
              for chunk, emb in st.session_state.vector_db]

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_n]

# ------------------------------------------------------
# Gemini generation
# ------------------------------------------------------
def ask_gemini(history, context, question):
    hist_text = "\n".join(
        [f"User: {h['user']}\nAssistant: {h['assistant']}" for h in history[-6:]]
    ) if history else "None"

    prompt = f"""
You must answer using the retrieved context AND the previous conversation.

Previous conversation:
{hist_text}

Retrieved context:
{chr(10).join([f"- {c}" for c in context])}

User question: {question}
"""
    model = genai.GenerativeModel(GEMINI_MODEL)
    response = model.generate_content(prompt)
    return response.text

# ------------------------------------------------------
# CHAT UI
# ------------------------------------------------------
if not st.session_state.sql_connected:
    st.info("Please configure SQL settings and click 'Connect to SQL & Initialize'.")
else:
    if len(st.session_state.dataset) == 0:
        st.info("Upload or load a CSV dataset to begin.")
    else:
        st.subheader("💬 Chat")

        user_query = st.chat_input("Ask something about the CSV data...")

        if user_query:
            retrieved = retrieve(user_query, st.session_state.history)
            context_chunks = [c for c, _ in retrieved]
            answer = ask_gemini(st.session_state.history, context_chunks, user_query)

            st.session_state.history.append({
                "user": user_query,
                "assistant": answer
            })

        for msg in st.session_state.history:
            with st.chat_message("user"):
                st.write(msg["user"])
            with st.chat_message("assistant"):
                st.write(msg["assistant"])
