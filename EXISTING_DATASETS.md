# 🚀 Working with Existing Datasets

If you already have datasets in SQL Server that aren't in the upload metadata, here are your options:

## Option 1: Using the Streamlit UI (Easiest)

### If your dataset is already tracked:

1. Go to **📂 Browse Datasets** in the sidebar
2. Select your dataset from the dropdown
3. You'll see a new button: **🔄 Generate Embeddings Now**
4. Click it to generate embeddings for that dataset
5. Wait for completion (shows progress with spinner)
6. Embeddings are saved automatically to:
   - `vector_store.index` (FAISS index)
   - `vector_map.json` (mapping file)
7. Then click **🚀 Work with This Dataset**
8. Open the chatbot and start asking questions!

## Option 2: Command Line Script (For Untracked Datasets)

If your dataset exists in SQL Server but isn't in the metadata table, use the `generate_embeddings.py` script:

### Step 1: Find your table name

Query SQL Server to see your tables:
```sql
SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES 
WHERE TABLE_SCHEMA = 'dbo'
```

### Step 2: Generate embeddings

```powershell
# With environment variable (preferred)
$env:GEMINI_API_KEY = "your-api-key"
python generate_embeddings.py your_table_name

# Or pass API key directly
python generate_embeddings.py your_table_name --api-key "your-api-key"

# With custom paths (optional)
python generate_embeddings.py your_table_name `
    --api-key "your-api-key" `
    --index-path "my_index.index" `
    --mapping-path "my_mapping.json"
```

### Example:
```powershell
$env:GEMINI_API_KEY = "AIza..."
python generate_embeddings.py dbo.Students
```

### Output:
```
🚀 Generating embeddings for table: dbo.Students
📁 Index path: vector_store.index
📁 Mapping path: vector_map.json
------------------------------------------------------------

✅ Dataset indexed successfully!
- Rows processed: 1000
- Embeddings created: 1000
- Embedding dimension: 768

📊 Statistics:
   table_name: dbo.Students
   total_rows: 1000
   total_chunks: 1000
   total_embeddings: 1000
   embedding_dimension: 768
   failed_embeddings: 0
   index_path: vector_store.index
   mapping_path: vector_map.json
```

## Option 3: First Upload to Track Dataset (Recommended)

For the best workflow with metadata tracking:

1. Go to **📤 Upload Dataset** in sidebar
2. Export your existing SQL Server data as CSV
3. Upload the CSV through the UI
4. Embeddings auto-generate (if API key configured)
5. Dataset is now tracked in metadata table
6. Use **📂 Browse Datasets** to select and chat

This way your datasets are properly tracked!

## 📊 Workflow Comparison

| Method | Tracked | Easy | Fast | Notes |
|--------|---------|------|------|-------|
| UI Upload | ✅ | ✅ | ✅ | Best for new data |
| UI Generate | ✅ | ✅ | ✅ | For existing tracked datasets |
| CLI Script | ❌ | ❌ | ✅ | For untracked datasets |

## 🆘 Troubleshooting

### "Table not found"
- Verify table name is correct: `SELECT * FROM your_table_name`
- Use fully qualified name: `dbo.TableName`

### "Invalid API key"
- Check Gemini API key at https://ai.google.dev
- Ensure key is correctly pasted (no extra spaces)
- Check API quota/limits

### "Embedding generation failed"
- Check SQL Server connection
- Verify table has data
- Check Gemini API rate limits
- Look at console output for specific error

### How to check if embeddings exist
```powershell
# Check in PowerShell
Test-Path "vector_store.index"  # Should be $true
Test-Path "vector_map.json"     # Should be $true
```

## 💡 Tips

1. **Large datasets**: Embedding generation scales with rows. 10K rows ≈ 2-5 minutes
2. **Reuse embeddings**: Once generated, they're cached locally for instant chatbot access
3. **Different datasets**: Each dataset needs its own FAISS index (name them differently)
4. **Track everything**: Use UI upload/generate for automatic tracking and easier management

## 🔄 Re-generating Embeddings

If you need to regenerate embeddings (e.g., if API changed):

```powershell
# Delete old files
Remove-Item "vector_store.index"
Remove-Item "vector_map.json"

# Then either:
# Option A: In UI - Click "Generate Embeddings Now"
# Option B: CLI - python generate_embeddings.py table_name --api-key "key"
```

---

**Next Steps:**
1. Choose your option above
2. Configure Gemini API key
3. Generate embeddings
4. Open chatbot and start asking questions!

Questions? Check the main README.md for more details.
