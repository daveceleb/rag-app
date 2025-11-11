"""
Common SQL queries for working with existing datasets.
Copy and run these in SQL Server Management Studio (SSMS) or SQL Server client.
"""

# ============================================================================
# 1. LIST ALL TABLES IN YOUR DATABASE
# ============================================================================
SELECT TABLE_NAME, TABLE_SCHEMA
FROM INFORMATION_SCHEMA.TABLES
WHERE TABLE_TYPE = 'BASE TABLE'
ORDER BY TABLE_NAME;


# ============================================================================
# 2. GET TABLE STRUCTURE AND ROW COUNT
# ============================================================================
-- For a specific table
SELECT 
    COLUMN_NAME,
    DATA_TYPE,
    IS_NULLABLE
FROM INFORMATION_SCHEMA.COLUMNS
WHERE TABLE_NAME = 'your_table_name'  -- Change this to your table name
ORDER BY ORDINAL_POSITION;

-- Get row count
SELECT COUNT(*) as RowCount FROM your_table_name;


# ============================================================================
# 3. PREVIEW DATA FROM TABLE
# ============================================================================
-- Show first 10 rows
SELECT TOP 10 * FROM your_table_name;

-- Show random sample
SELECT TOP 100 * FROM your_table_name 
ORDER BY NEWID();


# ============================================================================
# 4. CHECK METADATA TABLE (Created by RAG App)
# ============================================================================
SELECT * FROM dataset_metadata 
WHERE status = 'active'
ORDER BY upload_timestamp DESC;


# ============================================================================
# 5. EXPORT TABLE TO CSV (for uploading to UI)
# ============================================================================
-- In SQL Server Management Studio:
-- 1. Right-click on table
-- 2. Tasks → Export Data
-- 3. Choose CSV format
-- 4. Select destination


# ============================================================================
# 6. CREATE A TEST DATASET
# ============================================================================
-- If you want to test with sample data
CREATE TABLE TestEmployees (
    EmployeeID INT PRIMARY KEY,
    FirstName NVARCHAR(50),
    LastName NVARCHAR(50),
    Department NVARCHAR(50),
    Salary INT,
    HireDate DATE
);

INSERT INTO TestEmployees VALUES
(1, 'John', 'Smith', 'Sales', 50000, '2020-01-15'),
(2, 'Jane', 'Doe', 'IT', 65000, '2019-03-22'),
(3, 'Bob', 'Johnson', 'HR', 55000, '2018-06-10'),
(4, 'Alice', 'Williams', 'Sales', 52000, '2021-02-14'),
(5, 'Charlie', 'Brown', 'IT', 70000, '2017-11-03');


# ============================================================================
# 7. VIEW EXISTING DATASET METADATA
# ============================================================================
SELECT 
    table_name,
    file_name,
    upload_timestamp,
    row_count,
    column_count,
    columns_info,
    status
FROM dataset_metadata
ORDER BY upload_timestamp DESC;


# ============================================================================
# 8. DELETE A DATASET (Use with caution!)
# ============================================================================
-- Delete table
DROP TABLE [dataset_20251111_200000];  -- Change table name

-- Update metadata
DELETE FROM dataset_metadata 
WHERE table_name = 'dataset_20251111_200000';


# ============================================================================
# 9. COPY TABLE TO NEW NAME (for testing)
# ============================================================================
SELECT * INTO NewTableName FROM ExistingTableName;


# ============================================================================
# 10. GET COLUMN STATISTICS
# ============================================================================
SELECT 
    COLUMN_NAME,
    COUNT(*) as NonNullCount,
    COUNT(DISTINCT [COLUMN_NAME]) as UniqueValues
FROM your_table_name
GROUP BY COLUMN_NAME;


# ============================================================================
# PYTHON: LIST TABLES PROGRAMMATICALLY
# ============================================================================
"""
from database import get_engine
import pandas as pd

engine = get_engine()

# List all tables
query = """
SELECT TABLE_NAME 
FROM INFORMATION_SCHEMA.TABLES 
WHERE TABLE_TYPE = 'BASE TABLE'
"""
tables = pd.read_sql(query, con=engine)
print(tables)

# Get specific table info
table_name = "your_table_name"
df = pd.read_sql(f"SELECT TOP 100 * FROM [{table_name}]", con=engine)
print(f"Rows: {len(df)}, Columns: {len(df.columns)}")
print(df.dtypes)
"""


# ============================================================================
# PYTHON: EXPORT TABLE FOR UPLOAD
# ============================================================================
"""
from database import get_engine
import pandas as pd

engine = get_engine()
table_name = "your_table_name"

# Export to CSV
df = pd.read_sql(f"SELECT * FROM [{table_name}]", con=engine)
df.to_csv(f"{table_name}.csv", index=False)
print(f"✅ Exported {len(df)} rows to {table_name}.csv")

# Then upload the CSV through the RAG app UI
"""


# ============================================================================
# TIPS FOR WORKING WITH EXISTING DATA
# ============================================================================
"""
1. TABLE NAMING CONVENTIONS:
   - If table has special characters, use brackets: [Table Name]
   - Include schema: [dbo].[TableName]
   - Case-insensitive in SQL Server

2. DATA TYPES:
   - Text: NVARCHAR, VARCHAR, TEXT
   - Numbers: INT, BIGINT, FLOAT, DECIMAL
   - Dates: DATE, DATETIME, DATETIME2
   - All will be converted to text for embeddings

3. BEFORE GENERATING EMBEDDINGS:
   - Verify table has data (SELECT COUNT(*))
   - Check for large text columns (will be included)
   - Consider performance (embeddings ≈ 10K rows/min)

4. BEST PRACTICES:
   - Always backup your data first
   - Test with small tables first
   - Keep original table, work with copies if unsure
   - Document your table schemas
"""
