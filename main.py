import os
import httpx
from dotenv import load_dotenv
import json

def login():
    """
    Tests the Phoenix API login endpoint using credentials from the .env file.
    """
    print("Testing Phoenix API Login...")
    load_dotenv()

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
    load_dotenv()

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


def main():
    """
    Main function for the application.
    """
    # Call functions here as needed
    # login()
    queue_stats()

if __name__ == "__main__":
    main()
