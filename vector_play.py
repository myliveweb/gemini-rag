import os
# import httpx
from dotenv import load_dotenv
# import json
import psycopg2
from pgvector.psycopg2 import register_vector
# import torch
# from langchain_huggingface import HuggingFaceEmbeddings
# from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
import ollama
from docling.document_converter import DocumentConverter


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
                cur.execute("SELECT id, transcript_id, transcript_text, status, created_at FROM transcription_records ORDER BY id ASC LIMIT 10;")
                records = cur.fetchall()
                return records
    except Exception as e:
        print(f"Error fetching records: {e}")
        return None

def text_chunks(id: int, text: str):
    """
    Splits the input text into smaller chunks using RecursiveCharacterTextSplitter.
    """
    if not text:
        return []

    splitter = RecursiveCharacterTextSplitter(
        # chunk_size=128,  # Tiny size to force splitting
        # chunk_overlap=24,
        # chunk_size=256,  # Tiny size to force splitting
        # chunk_overlap=32,
        chunk_size=512,  # Small size to force splitting
        chunk_overlap=96,
        # chunk_size=768,  # Medium size to force splitting
        # chunk_overlap=128,
        # chunk_size=1024,  # Large size to force splitting
        # chunk_overlap=192,
        # length_function=len,
        # is_separator_regex=False,
        separators=["\n\n", "\n", " ", ""],  # Split hierarchy
    )

    chunks = splitter.split_text(text.strip())

    print(f"Original: {len(text.strip())} chars → {len(chunks)} chunks")
    # print(f"Full: {text}")

    # Show chunks
    for i, chunk in enumerate(chunks):
        print(f"ID: {id}, Chunk {i+1}: {chunk}")
        set_embeddings(id, chunk)

    return chunks

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
        records = get_list()
        if records:
            for sample_record in records:
                # sample_record = records[0] # Take the first record
                print(f"\nProcessing record with ID: {sample_record[0]}, Transcript ID: {sample_record[1]}")
                text_chunks(sample_record[0], sample_record[2]) # Pass transcript_text to text_chunks
        else:
            print("No records available to process for chunking.")

    except Exception as e:
        print(f"Ошибка подключения к базе данных или обработки: {e}")


# def test_embeddings():
#     """
#     A simple function to test embeddings.
#     """
#     print("Привет от Embeddings!")
#     try:
#         embeddings = HuggingFaceEmbeddings(
#             model_name='ai-sage/Giga-Embeddings-instruct',
#             encode_kwargs={},
#             model_kwargs={
#                 'device': 'cpu',
#                 'trust_remote_code': True,
#                 'model_kwargs': {'dtype': torch.bfloat16},
#                 'prompts': {'query': 'Instruct: Given a question, retrieve passages that answer the question\nQuery: '}
#             }
#         )

#         # Tokenizer
#         embeddings._client.tokenizer.tokenize("Hello world! I am GigaChat From Test")

#         # Query embeddings
#         query_embeddings = embeddings.embed_query("Hello world!!!")
#         print(f"Your embeddings: {query_embeddings[0:20]}...")
#         print(f"Vector size: {len(query_embeddings)}")

#         # Document embeddings
#         documents = ["foo bar", "bar foo"]
#         documents_embeddings = embeddings.embed_documents(documents)
#         print(f"Vector size: {len(documents_embeddings)} x {len(documents_embeddings[0])}")

#     except Exception as e:
#         print(f"Ошибка embeddings: {e}")


# def test_embeddings2():
#     """
#     A simple function to test embeddings 2.
#     """
#     print("Привет от Embeddings 2!")
#     try:
#         # Load the model
#         # We recommend enabling flash_attention_2 for better acceleration and memory saving
#         model = SentenceTransformer(
#             "ai-sage/Giga-Embeddings-instruct",
#             model_kwargs={
#                 # "attn_implementation": "flash_attention_2",
#                 "dtype": torch.bfloat16,
#                 "trust_remote_code": "True"
#             },
#             config_kwargs={
#                 "trust_remote_code": "True"
#             }
#         )
#         model.max_seq_length = 4096

#         # The queries and documents to embed
#         queries = [
#             'What is the capital of Russia?',
#             'Explain gravity'
#         ]
#         # No need to add instruction for retrieval documents
#         documents = [
#             # "The capital of Russia is Moscow.",
#             "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun."
#         ]

#         # Encode the queries and documents. Note that queries benefit from using a prompt
#         query_embeddings = model.encode(queries, prompt='Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: ')
#         document_embeddings = model.encode(documents)

#         # Compute the (cosine) similarity between the query and document embeddings
#         similarity = model.similarity(query_embeddings, document_embeddings)
#         print(similarity)
#         # tensor([[0.5846, 0.0702],
#         #         [0.0691, 0.6207]])

#     except Exception as e:
#         print(f"Ошибка embeddings: {e}")


def embeddings_ollama():
    """
    A simple function to test embeddings Ollama.
    """
    print("Привет от Embeddings Ollama!")
    try:
        text = "Привет от Ollama!"
        print(text)

        single = ollama.embed(
            model='bge-m3',
            input=text
        )
        print(len(text), len(single['embeddings'][0]), single['embeddings'][0][:20])  # vector length

    except Exception as e:
        print(f"Ошибка embeddings Ollama: {e}")


def set_embeddings(id: int, chunk: str):
    """
    A simple function to set embeddings Ollama.
    """
    try:
        # print(f"ID: {id}, Chunk: {chunk}")

        embedding = ollama.embed(
            model='bge-m3',
            input=chunk
        )

        with get_vector_db_connection() as con:
            with con.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO embeddings (document_id, content, embedding)
                    VALUES (%s, %s, %s);
                    """,
                    (id, chunk, embedding['embeddings'][0])
                )

    except Exception as e:
        print(f"Ошибка embeddings Ollama: {e}")



def test_docling():
    """
    A simple function to test Docling.
    """
    print("Привет от Docling!")
    try:
        text = "Привет от Docling!"
        print(text)

        # local file path (e.g., Path("/path/to/file.pdf")).
        source = "data/html/Leadslab.html"

        converter = DocumentConverter()
        result = converter.convert(source)

        # Print Markdown to stdout.
        print(result.document.export_to_markdown())

    except Exception as e:
        print(f"Ошибка Docling: {e}")


def search_vector(text: str):
    """
    A simple function Search Vector.
    """
    try:
        embedding = ollama.embed(
            model='bge-m3',
            input=text
        )

        with get_vector_db_connection() as con: # Changed to vector DB
            register_vector(con)
            with con.cursor() as cur:
                cur.execute("SELECT id, content, embedding <=> %s::vector AS score FROM embeddings ORDER BY score ASC;", (embedding['embeddings'][0],))
                records = cur.fetchall()

        n = 0
        for item in records:
            n += 1
            print(f"\n{n}. ID: {item[0]}, Score: {round(item[2], 3)}, {item[1]}")

    except Exception as e:
        print(f"Error fetching records: {e}")
        return None

if __name__ == "__main__":
    test_docling()
    # play_vector_script()
    # search_vector("сколько камер")
