import os
import re
import string
import time
from typing import Any, Dict, List, Literal, Optional
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
import torch
from dotenv import find_dotenv, load_dotenv
from markitdown import MarkItDown
from langchain_text_splitters import RecursiveCharacterTextSplitter
from collections import Counter
from loguru import logger
from langchain_chroma import Chroma
import json

from pathlib import Path

load_dotenv(find_dotenv())

start_time = time.time()

PUNCTUATION_PATTERN = re.compile(f"[{re.escape(string.punctuation)}]")
WHITESPACE_PATTERN = re.compile(r"\s+")

# Get the file path
output_folder = "documents"
# filename = "think_python_guide.pdf"
filename = "1.pdf"

file_path = Path(output_folder) / filename


def normalize_text(text: str) -> str:
    """Нормализация текста: удаление знаков препинания и специальных символов."""
    if not isinstance(text, str):
        raise ValueError("Входной текст должен быть строкой")

    # Удаление знаков препинания
    text = PUNCTUATION_PATTERN.sub(" ", text)
    # Удаление переносов строк и лишних пробелов
    text = WHITESPACE_PATTERN.sub(" ", text)
    # Приведение к нижнему регистру
    return text.lower().strip()


def get_markdown_content(file_path: Path) -> str:
    logger.info(f"Обработка PDF файла: {file_path} ...")
    # Initialize the converter
    md = MarkItDown()

    # Convert the Python guide to markdown
    result = md.convert(file_path)
    logger.info(f"Файл обработан за {time.time() - start_time:.2f} сек")
    return result.text_content


# Configure the text splitter with Q&A-optimized settings
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=os.getenv("MAX_CHUNK_SIZE"),         # Optimal chunk size for Q&A scenarios
    chunk_overlap=os.getenv("CHUNK_OVERLAP"),      # 20% overlap to preserve context
    separators=["\n\n", "\n", ". ", " ", ""],  # Split hierarchy
    length_function=len,
    is_separator_regex=False,
)


def process_document(doc, text_splitter):
    """Process a single document into chunks."""
    doc_chunks = text_splitter.split_text(doc["content"])
    return [{"content": normalize_text(chunk), "metadata": doc["metadata"]} for chunk in doc_chunks]


def get_documents(file_path: Path) -> dict[str, Any]:
    markdown_str = get_markdown_content(file_path)
    # Organize the converted document
    document = {
        'metadata': {
            'file_name': str(file_path),
            'category': 'books'
        },
        'text': markdown_str
    }

    return document


def split_text_into_chunks(text: str, metadata: Dict[str, Any]) -> List[Any]:
    """Разделение текста на чанки с сохранением метаданных."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=int(os.getenv("MAX_CHUNK_SIZE")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP")),
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
        is_separator_regex=False,
    )

    chunks = text_splitter.create_documents(texts=[text], metadatas=[metadata])
    return chunks


def generate_chroma_db(documents) -> Optional[Chroma]:
    """Инициализация ChromaDB."""
    try:
        # Создаем директорию для хранения базы данных, если она не существует
        os.makedirs(os.getenv("CHROMA_PATH"), exist_ok=True)

        if not documents:
            logger.warning("Нет документов для добавления в базу данных")
            return None

        # Инициализируем модель эмбеддингов
        embeddings = HuggingFaceEmbeddings(
            model_name=os.getenv("EMBEDDING_MODEL_NAME"),
            model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        # Подготавливаем данные для Chroma
        all_chunks = []
        for i, doc in enumerate(documents):
            chunks = split_text_into_chunks(doc["text"], doc["metadata"])
            all_chunks.extend(chunks)
            logger.info(
                f"Документ {i+1}/{len(documents)} разбит на {len(chunks)} чанков"
            )

        # Создаем векторное хранилище
        texts = [normalize_text(chunk.page_content) for chunk in all_chunks]
        metadatas = [chunk.metadata for chunk in all_chunks]
        ids = [f"doc_{i}" for i in range(len(all_chunks))]

        chroma_db = Chroma.from_texts(
            texts=texts,
            embedding=embeddings,
            ids=ids,
            metadatas=metadatas,
            persist_directory=os.getenv("CHROMA_PATH"),
            collection_name=os.getenv("CHROMA_COLLECTION_NAME"),
            collection_metadata={
                "hnsw:space": "cosine",
            },
        )

        logger.success(
            f"База Chroma инициализирована, добавлено {len(all_chunks)} чанков из {len(documents)} документов"
        )
        return chroma_db
    except Exception as e:
        logger.error(f"Ошибка инициализации Chroma: {e}")
        raise


def main():
    logger.success("Старт конвертер")

    processed_document = get_documents(file_path)
    documents = [processed_document]

    generate_chroma_db(documents)

    # Process all documents and create chunks
    # all_chunks = []
    # for doc in documents:
    #     doc_chunks = process_document(doc, text_splitter)
    #     all_chunks.extend(doc_chunks)

    # source_counts = Counter(chunk["metadata"]["file"] for chunk in all_chunks)
    # chunk_lengths = [len(chunk["content"]) for chunk in all_chunks]

    # print(f"Total chunks created: {len(all_chunks)}")
    # print(f"Chunk length: {min(chunk_lengths)}-{max(chunk_lengths)} characters")
    # print(f"Source document: {Path(documents[0]['metadata']["file"]).name}")

    # # Show chunks
    # i = 1
    # for chunk in all_chunks[100:120]:
    #     print(f"Chunk {i}: {chunk}")
    #     i += 1

    logger.success(f"Время исполнения: {time.time() - start_time:.2f} сек")


if __name__ == "__main__":
    main()