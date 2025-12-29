import os
import httpx
from dotenv import load_dotenv
import json
import psycopg2
from langchain_text_splitters import RecursiveCharacterTextSplitter

print("--- Script execution started ---")
load_dotenv()

def get_vector_db_connection():
    """Establishes a connection to the PostgreSQL database with vector support."""
    vector_db_url = os.getenv("VECTOR_DATABASE_URL")
    if not vector_db_url:
        raise Exception("VECTOR_DATABASE_URL environment variable not set.")
    return psycopg2.connect(vector_db_url)

def get_list():
    """
    Fetches all records from the transcription_records table, sorted by id ASC.
    Returns a list of tuples (id, transcript_id, transcript_text, status, created_at) or None.
    """
    print("Fetching all records from transcription_records table...")
    try:
        with get_vector_db_connection() as con: # Changed to vector DB
            with con.cursor() as cur:
                cur.execute("SELECT id, transcript_id, transcript_text, status, created_at FROM transcription_records ORDER BY id ASC;")
                records = cur.fetchall()
                return records
    except Exception as e:
        print(f"Error fetching records: {e}")
        return None

def text_chunks(text: str):
    """
    Splits the input text into smaller chunks using RecursiveCharacterTextSplitter.
    """
    if not text:
        return []

    demo_splitter = RecursiveCharacterTextSplitter(
        chunk_size=150,  # Small size to force splitting
        chunk_overlap=30,
        separators=["\n\n", "\n", ". ", " ", ""],  # Split hierarchy
    )

    sample_chunks = demo_splitter.split_text(text.strip())

    print(f"Original: {len(text.strip())} chars → {len(sample_chunks)} chunks")

    # Show chunks
    for i, chunk in enumerate(sample_chunks):
        print(f"Chunk {i+1}: {chunk}")
    
    return sample_chunks

def test_qdrant_connection():
    """
    Tests the connection to Qdrant and retrieves a list of collections.
    """
    print("Testing Qdrant connection...")
    try:
        # client = QdrantClient(url="http://localhost:6333") # Qdrant removed
        print("ChromaDB client initialized successfully (in-memory).")
        
        # collections = client.get_collections()
        # print("Successfully connected to Qdrant!")
        # print("Collections:", collections)
        print("QdrantClient removed. No test executed.")
        
    except Exception as e:
        print(f"Error connecting to Qdrant or fetching collections: {e}")

def create_vector_transcription_records_table():
    """
    Creates the transcription_records table in the vector PostgreSQL database.
    """
    print("Creating transcription_records table in vector database...")
    try:
        with get_vector_db_connection() as con:
            with con.cursor() as cur:
                cur.execute('''
                    CREATE TABLE IF NOT EXISTS transcription_records (
                        id SERIAL PRIMARY KEY,
                        transcript_id TEXT NOT NULL UNIQUE,
                        transcript_text TEXT,
                        status VARCHAR(50),
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
        print("Table transcription_records created successfully in vector database (if it didn't exist).")
    except psycopg2.Error as e:
        print(f"Database error: {e}")

def sync_data_to_vector_db():
    """
    Fetches all records from the main DB and syncs them to the vector DB.
    """
    print("Starting data sync to vector DB...")
    
    # 1. Fetch all records from the main database
    records = get_list()
    if not records:
        print("No records found in main database to sync.")
        return

    print(f"Found {len(records)} records in main DB to sync.")

    # 2. Connect to the vector database and insert/update records
    try:
        with get_vector_db_connection() as con:
            with con.cursor() as cur:
                for record in records:
                    # record is a tuple: (id, transcript_id, transcript_text, status, created_at)
                    # We only need to insert transcript_id, transcript_text, status, as id and created_at are auto-generated
                    transcript_id = record[1]
                    transcript_text = record[2]
                    status = record[3]
                    
                    cur.execute(
                        """
                        INSERT INTO transcription_records (transcript_id, transcript_text, status)
                        VALUES (%s, %s, %s)
                        ON CONFLICT (transcript_id) DO UPDATE SET
                            transcript_text = EXCLUDED.transcript_text,
                            status = EXCLUDED.status,
                            created_at = CURRENT_TIMESTAMP;
                        """,
                        (transcript_id, transcript_text, status)
                    )
        print("Successfully synced data to the vector database!")
    except psycopg2.Error as e:
        print(f"Database error during sync: {e}")


def play_vector_script():
    """
    A simple function to test connection and output a greeting.
    """
    print("Привет от Джина!")
    try:
        # Test vector database connection
        with get_vector_db_connection() as con_vector:
            print("Успешно подключились к векторной базе данных PostgreSQL!")
            with con_vector.cursor() as cur_vector:
                cur_vector.execute("SELECT version();")
                vector_db_version = cur_vector.fetchone()[0]
                print(f"Версия векторной PostgreSQL: {vector_db_version}")
        
        # create_vector_transcription_records_table()
        # sync_data_to_vector_db()
        # test_chroma_connection() # Uncomment to test ChromaDB connection

        # Get one transcription record
        # records = get_list()
        # if records:
        #     sample_record = records[0] # Take the first record
        #     print(f"\nProcessing record with ID: {sample_record[0]}, Transcript ID: {sample_record[1]}")
        #     text_chunks(sample_record[2]) # Pass transcript_text to text_chunks
        # else:
        #     print("No records available to process for chunking.")

    except Exception as e:
        print(f"Ошибка подключения к базе данных или обработки: {e}")
if __name__ == "__main__":
    play_vector_script()
