import time
from langchain_huggingface import HuggingFaceEmbeddings
import torch
from loguru import logger
from langchain_chroma import Chroma
import json
from langchain_docling.loader import DoclingLoader # Отказ
from langchain_opendataloader_pdf import OpenDataLoaderPDFLoader # Мультимодальная загрузка 45 сек

import requests
from pathlib import Path

from markitdown import MarkItDown # Мульти? 17 сек


CHROMA_PATH = "./next_chroma_db"
COLLECTION_NAME = "next_data"

# Get the file path
output_folder = "documents"
filename = "think_python_guide.pdf"
url = "https://greenteapress.com/thinkpython/thinkpython.pdf"
file_path = Path(output_folder) / filename

FILE_PATH = Path(output_folder) / "2.pdf"

def download_file(url: str, file_path: Path):
    response = requests.get(url, stream=True, timeout=30)
    response.raise_for_status()
    file_path.write_bytes(response.content)


# Download the file if it doesn't exist
if not file_path.exists():
    download_file(
        url=url,
        file_path=file_path,
    )

start_time = time.time()

logger.info("Загрузка PDF файла...")

# Initialize the converter
md = MarkItDown()

# file_path = "2.pdf"

# Convert the Python guide to markdown
result = md.convert(file_path)
python_guide_content = result.text_content

# Display the conversion results
print("First 300 characters:")
print(python_guide_content[:1000] + "...")

# Список данных о товарах

# loader = DoclingLoader(file_path=FILE_PATH)
# documents = loader.load()

# print(json.dumps(documents[200], indent=4, ensure_ascii=False))

# loader = OpenDataLoaderPDFLoader(
#     file_path=FILE_PATH,
#     format="markdown"
# )
# documents = loader.load()


# print("First 300 characters:")

# for doc in documents:
#     print(doc.metadata, doc.page_content[:300])

logger.info(f"Файл загружен за {time.time() - start_time:.2f} сек")

# Сборка итогового списка документов с автоинкрементом ID
# documents = []
# for index, item in enumerate(raw_data, start=1):
#     doc = {
#         "text": item["text"],
#         "metadata": {
#             "id": str(index),
#             "type": "product",
#             "category": item["category"],
#             "price": item["price"],
#             "stock": item["stock"]
#         }
#     }
#     documents.append(doc)

# def generate_chroma_db():
#     try:
#         start_time = time.time()
        
#         logger.info("Загрузка модели эмбеддингов...")
#         embeddings = HuggingFaceEmbeddings(
#             model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
#             model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
#             encode_kwargs={"normalize_embeddings": True},
#         )
#         logger.info(f"Модель загружена за {time.time() - start_time:.2f} сек")
        
#         logger.info("Создание Chroma DB...")
#         chroma_db = Chroma.from_texts(
#             texts=[item["text"] for item in SHOP_DATA],
#             embedding=embeddings,
#             ids=[str(item["metadata"]["id"]) for item in SHOP_DATA],
#             metadatas=[item["metadata"] for item in SHOP_DATA],
#             persist_directory=CHROMA_PATH,
#             collection_name=COLLECTION_NAME,
#         )
#         logger.info(f"Chroma DB создана за {time.time() - start_time:.2f} сек")
        
#         return chroma_db
#     except Exception as e:
#         logger.error(f"Ошибка: {e}")
#         raise

def main():
    logger.success("Старт")

# Вывод результата
if __name__ == "__main__":
    main()