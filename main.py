import os
import httpx
from tqdm import tqdm

def download_files_from_list(file_path, download_dir):
    """
    Downloads files from a list of URLs in a text file.

    Args:
        file_path (str): The path to the text file containing the URLs.
        download_dir (str): The directory where the files will be saved.
    """
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)

    try:
        with open(file_path, 'r') as f:
            urls = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return

    if not urls:
        print("No URLs found in the file.")
        return

    with httpx.Client() as client:
        for url in tqdm(urls, desc="Downloading files"):
            try:
                response = client.get(url, follow_redirects=True)
                response.raise_for_status()  # Raise an exception for bad status codes

                # Extract filename from URL
                filename = os.path.join(download_dir, url.split('/')[-1])

                with open(filename, 'wb') as f:
                    f.write(response.content)

                tqdm.write(f"Downloaded {url} to {filename}")

            except httpx.RequestError as exc:
                tqdm.write(f"An error occurred while requesting {exc.request.url!r}: {exc}")
            except httpx.HTTPStatusError as exc:
                tqdm.write(f"Error response {exc.response.status_code} while requesting {exc.request.url!r}.")


def main():
    """
    Main function to initiate the download process.
    """
    links_file = 'data/html/Link.txt'
    download_folder = 'data/mp3'
    
    print(f"Starting download of files from {links_file} to {download_folder}")
    download_files_from_list(links_file, download_folder)
    print("All downloads completed.")


if __name__ == "__main__":
    main()