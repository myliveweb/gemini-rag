import os
import httpx
from dotenv import load_dotenv
import json
import psycopg2

print("--- Script execution started ---")
load_dotenv() # Переместил сюда

def get_db_connection():
    """Establishes a connection to the PostgreSQL database."""
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise Exception("DATABASE_URL environment variable not set.")
    return psycopg2.connect(db_url)

def login():
    """
    Tests the Phoenix API login endpoint using credentials from the .env file.
    """
    print("Testing Phoenix API Login...")

    username = os.getenv("USERNAME")
    password = os.getenv("PASSWORD")
    base_url = os.getenv("PHOENIX_API_BASE_URL")
    
    if not username or not password or not base_url:
        print("Error: USERNAME, PASSWORD, and PHOENIX_API_BASE_URL must be set in the .env file.")
        return

    url = f"{base_url}/auth/login"
    headers = {"Content-Type": "application/json"}
    data = {"username": username, "password": password}

    try:
        with httpx.Client() as client:
            response = client.post(url, headers=headers, data=json.dumps(data))
            response.raise_for_status()
            
            print("API Response:")
            print(response.text)

    except httpx.RequestError as exc:
        print(f"An error occurred while requesting {exc.request.url!r}: {exc}")
    except httpx.HTTPStatusError as exc:
        print(f"Error response {exc.response.status_code} while requesting {exc.request.url!r}:")
        print(response.text)
    print("API login test finished.")

def queue_stats():
    """
    Fetches queue statistics from the Phoenix API using API key authorization.
    """
    print("Fetching queue statistics...")

    api_key = os.getenv("PHOENIX_API_KEY")
    base_url = os.getenv("PHOENIX_API_BASE_URL")

    if not api_key or not base_url:
        print("Error: PHOENIX_API_KEY and PHOENIX_API_BASE_URL must be set in the .env file.")
        return

    url = f"{base_url}/admin/queue/stats"
    headers = {"X-API-Key": api_key}

    try:
        with httpx.Client() as client:
            response = client.get(url, headers=headers)
            response.raise_for_status()
            
            print("Queue Stats API Response:")
            print(response.text)

    except httpx.RequestError as exc:
        print(f"An error occurred while requesting {exc.request.url!r}: {exc}")
    except httpx.HTTPStatusError as exc:
        print(f"Error response {exc.response.status_code} while requesting {exc.request.url!r}.")
    print("Queue stats fetch finished.")


def setup_database():
    """
    Connects to the PostgreSQL database and creates the transcriptions table if it doesn't exist.
    """
    print("Setting up database...")
    try:
        with get_db_connection() as con:
            with con.cursor() as cur:
                cur.execute('''
                    CREATE TABLE IF NOT EXISTS transcriptions (
                        id SERIAL PRIMARY KEY,
                        filename TEXT NOT NULL UNIQUE,
                        url TEXT NOT NULL,
                        status TEXT NOT NULL DEFAULT 'downloaded'
                    )
                ''')
        print("Database setup complete.")
    except psycopg2.Error as e:
        print(f"Database error: {e}")

def populate_transcriptions():
    """
    Clears the transcriptions table, then populates it with data from downloaded mp3 files.
    Files larger than 400KB are deleted, and their corresponding links are removed from Link.txt.
    """
    print("Populating transcriptions table...")
    links_file = 'data/html/Link.txt'
    mp3_dir = 'data/mp3'
    max_size_kb = 400
    max_size_bytes = max_size_kb * 1024
    
    urls_to_remove = []

    try:
        # 1. Read original URLs
        with open(links_file, 'r') as f:
            urls = [line.strip() for line in f if line.strip()]
        
        url_map = {url.split('/')[-1]: url for url in urls}
        
        # 2. Connect to DB and clear the table
        with get_db_connection() as con:
            with con.cursor() as cur:
                print("Clearing transcriptions table...")
                cur.execute("TRUNCATE TABLE transcriptions RESTART IDENTITY;")
                
                # 3. Process files
                files = os.listdir(mp3_dir)
                for filename in files:
                    if not filename.endswith(".mp3"):
                        continue

                    file_path = os.path.join(mp3_dir, filename)
                    file_size = os.path.getsize(file_path)

                    if file_size > max_size_bytes:
                        print(f"File {filename} is too large ({file_size / 1024:.2f} KB). Deleting...")
                        os.remove(file_path)
                        if filename in url_map:
                            urls_to_remove.append(url_map[filename])
                    elif filename in url_map:
                        # File is of valid size, add to DB
                        cur.execute(
                            "INSERT INTO transcriptions (filename, url) VALUES (%s, %s) ON CONFLICT (filename) DO NOTHING",
                            (filename, url_map[filename])
                        )
                    else:
                        print(f"Warning: URL not found for {filename}. Skipping database entry.")

        print("Transcriptions table populated.")

        # 4. Update Link.txt if necessary
        if urls_to_remove:
            print(f"Removing {len(urls_to_remove)} oversized file links from {links_file}...")
            # Create a set for efficient lookup
            urls_to_remove_set = set(urls_to_remove)
            # Read original lines again
            with open(links_file, 'r') as f:
                original_lines = f.readlines()
            # Filter out the lines to be removed
            updated_lines = [line for line in original_lines if line.strip() not in urls_to_remove_set]
            # Write the updated content back
            with open(links_file, 'w') as f:
                f.writelines(updated_lines)
            print("Link.txt updated.")

    except FileNotFoundError:
        print(f"Error: {links_file} or {mp3_dir} not found.")
    except (psycopg2.Error, OSError) as e:
        print(f"An error occurred: {e}")

def upload_transcription(file_path):
    """
    Uploads an audio file to the /transcription/upload endpoint using API key authorization.
    """
    print(f"Uploading file: {file_path}")

    api_key = os.getenv("PHOENIX_API_KEY")
    base_url = os.getenv("PHOENIX_API_BASE_URL")

    if not api_key or not base_url:
        print("Error: PHOENIX_API_KEY and PHOENIX_API_BASE_URL must be set in the .env file.")
        return

    url = f"{base_url}/transcription/upload"
    headers = {"X-API-Key": api_key}

    try:
        with open(file_path, 'rb') as f:
            files = {'audio': (os.path.basename(file_path), f, 'audio/mpeg')}
            with httpx.Client() as client:
                response = client.post(url, headers=headers, files=files)
                response.raise_for_status()
                
                print("API Response:")
                print(response.text)

    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
    except httpx.RequestError as exc:
        print(f"An error occurred while requesting {exc.request.url!r}: {exc}")
    except httpx.HTTPStatusError as exc:
        print(f"Error response {exc.response.status_code} while requesting {exc.request.url!r}:")
        print(response.text)
    print("File upload finished.")


def list_transcriptions(filters=None):
    """
    Fetches a list of transcriptions from the Phoenix API, with optional filters.
    Authorization is done via API key. Returns a list of jobs or None.
    """
    print("Fetching list of transcriptions...")

    api_key = os.getenv("PHOENIX_API_KEY")
    base_url = os.getenv("PHOENIX_API_BASE_URL")

    if not api_key or not base_url:
        print("Error: PHOENIX_API_KEY and PHOENIX_API_BASE_URL must be set in the .env file.")
        return None

    url = f"{base_url}/transcription/list"
    headers = {"X-API-Key": api_key}
    
    params = {k: v for k, v in filters.items() if v is not None} if filters else {}

    try:
        with httpx.Client() as client:
            response = client.get(url, headers=headers, params=params)
            response.raise_for_status()
            
            print("Transcription List API Response (Status Code:", response.status_code, "):")
            try:
                parsed_json = response.json()
                if "jobs" in parsed_json and isinstance(parsed_json["jobs"], list):
                    return parsed_json["jobs"]
                else:
                    print("No jobs found or unexpected response structure.")
                    print(json.dumps(parsed_json, indent=2))
                    return None
            except json.JSONDecodeError:
                print("Could not decode JSON response.")
                print(response.text)
                return None

    except httpx.RequestError as exc:
        print(f"An error occurred while requesting {exc.request.url!r}: {exc}")
    except httpx.HTTPStatusError as exc:
        print(f"Error response {exc.response.status_code} while requesting {exc.request.url!r}:")
        print(response.text)
    print("Transcription list fetch finished.")
    return None


def delete_transcription(transcription_id: str):
    """
    Deletes a transcription by its ID from the Phoenix API.
    Authorization is done via API key.
    """
    print(f"Attempting to delete transcription with ID: {transcription_id}...")

    api_key = os.getenv("PHOENIX_API_KEY")
    base_url = os.getenv("PHOENIX_API_BASE_URL")

    if not api_key or not base_url:
        print("Error: PHOENIX_API_KEY and PHOENIX_API_BASE_URL must be set in the .env file.")
        return
    
    if not transcription_id:
        print("Error: Transcription ID must be provided.")
        return

    url = f"{base_url}/transcription/{transcription_id}"
    headers = {"X-API-Key": api_key}

    try:
        with httpx.Client() as client:
            response = client.delete(url, headers=headers)
            response.raise_for_status()
            
            if response.status_code == 204:
                print(f"Successfully deleted transcription ID: {transcription_id}. Mission accomplished!")
            elif response.status_code == 200:
                print(f"Successfully deleted transcription ID: {transcription_id}. Response:")
                print(response.text)
            else:
                print(f"Received status code {response.status_code}. Response:")
                print(response.text)

    except httpx.RequestError as exc:
        print(f"An error occurred while requesting {exc.request.url!r}: {exc}")
    except httpx.HTTPStatusError as exc:
        print(f"Error response {exc.response.status_code} while requesting {exc.request.url!r}:")
        print(response.text)
    print("Transcription deletion attempt finished.")

def create_transcription_records_table():
    """
    Creates the transcription_records table in the PostgreSQL database.
    """
    print("Creating transcription_records table...")
    try:
        with get_db_connection() as con:
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
        print("Table transcription_records created successfully (if it didn't exist).")
    except psycopg2.Error as e:
        print(f"Database error: {e}")


def sync_transcriptions_to_db():
    """
    Fetches all transcriptions from the API and syncs them to the transcription_records table.
    """
    print("Starting transcription sync to DB...")
    
    # Fetch all transcriptions (no filters)
    all_jobs = list_transcriptions()

    if not all_jobs:
        print("No transcriptions found to sync.")
        return

    records_to_sync = []
    for job in all_jobs:
        job_id = job.get("id")
        status = job.get("status")
        transcript_raw = job.get("transcript")
        transcript_text = None
        
        if transcript_raw:
            try:
                # The transcript is a JSON string, so we need to parse it
                transcript_json = json.loads(transcript_raw)
                transcript_text = transcript_json.get("text")
            except (json.JSONDecodeError, TypeError):
                 # If it's not a valid JSON or not a string, we can fall back to the raw content
                transcript_text = str(transcript_raw)
        
        if job_id:
            records_to_sync.append((job_id, transcript_text, status))

    if not records_to_sync:
        print("No valid records to sync.")
        return

    print(f"Found {len(records_to_sync)} records to sync.")

    try:
        with get_db_connection() as con:
            with con.cursor() as cur:
                for record in records_to_sync:
                    cur.execute(
                        """
                        INSERT INTO transcription_records (transcript_id, transcript_text, status)
                        VALUES (%s, %s, %s)
                        ON CONFLICT (transcript_id) DO UPDATE SET
                            transcript_text = EXCLUDED.transcript_text,
                            status = EXCLUDED.status,
                            created_at = CURRENT_TIMESTAMP;
                        """,
                        record
                    )
        print("Successfully synced transcriptions to the database!")
    except psycopg2.Error as e:
        print(f"Database error during sync: {e}")

def update_status_to_pending():
    """
    Updates the status of all records in the transcription_records table to 'pending'.
    """
    print("Updating transcription records status to 'pending'...")
    try:
        with get_db_connection() as con:
            with con.cursor() as cur:
                cur.execute(
                    """
                    UPDATE transcription_records
                    SET status = 'pending';
                    """
                )
        print("Successfully updated all transcription records status to 'pending'!")
    except psycopg2.Error as e:
        print(f"Database error during status update: {e}")


def main():
    """
    Main function for the application.
    """
    print("--- main() function called ---")
    # setup_database()
    # populate_transcriptions()
    # login()
    # queue_stats()
    # create_transcription_records_table()
    # sync_transcriptions_to_db()
    # update_status_to_pending()

    # --- List transcriptions example ---
    # Define filters. Use None for parameters you don't want to send.
    # transcription_filters = {
    #     "page": 1,
    #     "limit": 1,
    #     "sort_by": "created_at",
    #     "f_order": "desc",
    #     "status": None,
    #     "q": None,
    #     "updated_after": None
    # }
    # list_transcriptions(filters=transcription_filters)

    # --- Delete transcription example ---
    # test_id_to_delete = "78887e30-18c3-4413-b0a7-68df0711fea8"
    # delete_transcription(test_id_to_delete)

    # --- Upload transcription example ---
    # mp3_dir = 'data/mp3'
    # if os.path.exists(mp3_dir):
    #     files = [f for f in os.listdir(mp3_dir) if f.endswith(".mp3")]
    #     if files:
    #         sample_file_path = os.path.join(mp3_dir, files[0])
    #         # upload_transcription(sample_file_path)
    #     else:
    #         print(f"No mp3 files found in {mp3_dir}.")
if __name__ == "__main__":
    main()