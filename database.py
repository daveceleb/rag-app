"""
Database module for handling SQL Server connections and operations.
"""

import pandas as pd
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.exc import SQLAlchemyError
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# SQL Server Configuration
SERVER = "DESKTOP-4EBCN4A\\SQLEXPRESS"
DATABASE = "AcademicDB"
DRIVER = "ODBC Driver 17 for SQL Server"


def get_connection_string():
    """Generate SQL Server connection string."""
    connection_string = f"mssql+pyodbc://{SERVER}/{DATABASE}?driver={DRIVER}"
    return connection_string


def get_engine():
    """Create and return SQLAlchemy engine."""
    try:
        engine = create_engine(get_connection_string(), echo=False)
        return engine
    except Exception as e:
        logger.error(f"Failed to create engine: {str(e)}")
        raise


def test_connection():
    """Test connection to SQL Server."""
    try:
        engine = get_engine()
        with engine.connect() as connection:
            result = connection.execute(text("SELECT 1"))
            logger.info("✅ Connected to SQL Server successfully")
            return True, "Connected to SQL Server"
    except SQLAlchemyError as e:
        error_msg = f"SQL Connection Failed: {str(e)}"
        logger.error(error_msg)
        return False, error_msg
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        logger.error(error_msg)
        return False, error_msg


def create_metadata_table():
    """Create a metadata table to track uploaded datasets if it doesn't exist."""
    try:
        engine = get_engine()
        with engine.connect() as connection:
            # Create metadata table
            create_table_query = """
            IF NOT EXISTS (SELECT * FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = 'dataset_metadata')
            CREATE TABLE dataset_metadata (
                id INT PRIMARY KEY IDENTITY(1,1),
                table_name NVARCHAR(255) NOT NULL UNIQUE,
                file_name NVARCHAR(255) NOT NULL,
                upload_timestamp DATETIME DEFAULT GETDATE(),
                row_count INT NOT NULL,
                column_count INT NOT NULL,
                columns_info NVARCHAR(MAX),
                status NVARCHAR(50) DEFAULT 'active'
            )
            """
            connection.execute(text(create_table_query))
            connection.commit()
            logger.info("Metadata table created or already exists")
            return True
    except Exception as e:
        logger.error(f"Failed to create metadata table: {str(e)}")
        return False


def save_dataset_to_sql(df, table_name, original_filename):
    """
    Save a pandas DataFrame to SQL Server.
    
    Args:
        df (pd.DataFrame): DataFrame to save
        table_name (str): Target table name in SQL Server
        original_filename (str): Original CSV filename
        
    Returns:
        tuple: (success: bool, message: str, row_count: int)
    """
    try:
        engine = get_engine()
        
        # Ensure metadata table exists
        create_metadata_table()
        
        # Write DataFrame to SQL Server
        df.to_sql(table_name, con=engine, if_exists='replace', index=False)
        
        row_count = len(df)
        column_count = len(df.columns)
        columns_info = ", ".join([f"{col} ({str(df[col].dtype)})" for col in df.columns])
        
        # Store metadata
        with engine.connect() as connection:
            insert_metadata_query = f"""
            INSERT INTO dataset_metadata (table_name, file_name, row_count, column_count, columns_info)
            VALUES ('{table_name}', '{original_filename}', {row_count}, {column_count}, N'{columns_info}')
            """
            connection.execute(text(insert_metadata_query))
            connection.commit()
        
        message = f"✅ Dataset '{original_filename}' successfully saved to database table '{table_name}'"
        logger.info(f"{message} | Rows: {row_count}")
        
        return True, message, row_count
    
    except SQLAlchemyError as e:
        error_msg = f"Database error while saving dataset: {str(e)}"
        logger.error(error_msg)
        return False, error_msg, 0
    except Exception as e:
        error_msg = f"Unexpected error while saving dataset: {str(e)}"
        logger.error(error_msg)
        return False, error_msg, 0


def get_dataset_metadata():
    """Retrieve all uploaded datasets metadata."""
    try:
        engine = get_engine()
        query = "SELECT * FROM dataset_metadata WHERE status = 'active' ORDER BY upload_timestamp DESC"
        df = pd.read_sql(query, con=engine)
        return df
    except Exception as e:
        logger.error(f"Failed to retrieve metadata: {str(e)}")
        return pd.DataFrame()


def get_table_structure(table_name):
    """Get the structure and preview of a table."""
    try:
        engine = get_engine()
        with engine.connect() as connection:
            # Get column info
            inspector = inspect(engine)
            columns = inspector.get_columns(table_name)
            
            # Get sample data
            query = f"SELECT TOP 5 * FROM [{table_name}]"
            preview_df = pd.read_sql(query, con=engine)
            
            return columns, preview_df
    except Exception as e:
        logger.error(f"Failed to get table structure: {str(e)}")
        return None, None


def delete_dataset(table_name):
    """Delete a dataset table and its metadata."""
    try:
        engine = get_engine()
        with engine.connect() as connection:
            # Delete table
            drop_table_query = f"DROP TABLE IF EXISTS [{table_name}]"
            connection.execute(text(drop_table_query))
            
            # Update metadata
            update_metadata_query = f"UPDATE dataset_metadata SET status = 'deleted' WHERE table_name = '{table_name}'"
            connection.execute(text(update_metadata_query))
            
            connection.commit()
            logger.info(f"Dataset '{table_name}' deleted successfully")
            return True
    except Exception as e:
        logger.error(f"Failed to delete dataset: {str(e)}")
        return False
