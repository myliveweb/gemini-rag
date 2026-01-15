import os
import re
import string
import time
import shutil

# Замер времени начала загрузки
script_load_start = time.time()

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

PUNCTUATION_PATTERN = re.compile(f"[{re.escape(string.punctuation)}]")
WHITESPACE_PATTERN = re.compile(r"\s+")

# Флаг для очистки существующей базы данных
CLEAR_EXISTING_DB = True

# Get the file path
output_folder = "documents"
# filename = "think_python_guide.pdf"
filename = "1.pdf"

file_path = Path(output_folder) / filename

# Замер времени окончания загрузки и начальной настройки
script_load_end = time.time()
logger.info(f"Загрузка скрипта и импортов: {script_load_end - script_load_start:.2f} сек")


def normalize_text(text: str) -> str:
    """
    Нормализует текст, удаляя знаки препинания, лишние пробелы и приводя его к нижнему регистру.
    Это важный шаг для подготовки текста к векторизации и семантическому поиску.

    Args:
        text (str): Входной текст для нормализации.

    Returns:
        str: Нормализованный текст.

    Raises:
        ValueError: Если входные данные не являются строкой.
    """
    if not isinstance(text, str):
        raise ValueError("Входной текст должен быть строкой")

    # Удаление знаков препинания
    text = PUNCTUATION_PATTERN.sub(" ", text)
    # Удаление переносов строк и лишних пробелов
    text = WHITESPACE_PATTERN.sub(" ", text)
    # Приведение к нижнему регистру
    return text.lower().strip()


def get_markdown_content(file_path: Path) -> str:
    """
    Конвертирует содержимое PDF-файла в формат Markdown с помощью библиотеки MarkItDown.
    Markdown является предпочтительным форматом для дальнейшей обработки текста в LLM-пайплайнах.

    Args:
        file_path (Path): Путь к PDF-файлу.

    Returns:
        str: Содержимое файла в формате Markdown.
    """
    start_time = time.time()
    logger.info(f"Обработка PDF файла: {file_path} ...")
    # Initialize the converter
    md = MarkItDown()

    # Convert the Python guide to markdown
    result = md.convert(file_path)
    end_time = time.time()
    logger.info(f"PDF обработан за {end_time - start_time:.2f} сек")
    return result.text_content


# Configure the text splitter with Q&A-optimized settings
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=os.getenv("MAX_CHUNK_SIZE"),  # Optimal chunk size for Q&A scenarios
    chunk_overlap=os.getenv("CHUNK_OVERLAP"),  # 20% overlap to preserve context
    separators=["\n\n", "\n", ". ", " ", ""],  # Split hierarchy
    length_function=len,
    is_separator_regex=False,
)


def process_document(doc, text_splitter):
    """
    Обрабатывает один документ, разделяя его на части (чанки), оптимизированные для LLM.
    Разделение на чанки необходимо для того, чтобы не превышать лимит токенов модели.

    Args:
        doc (dict): Словарь с контентом и метаданными документа.
        text_splitter: Экземпляр сплиттера текста.

    Returns:
        list: Список чанков с нормализованным контентом и метаданными.
    """
    doc_chunks = text_splitter.split_text(doc["content"])
    return [
        {"content": normalize_text(chunk), "metadata": doc["metadata"]}
        for chunk in doc_chunks
    ]


def get_documents(file_path: Path) -> dict[str, Any]:
    """
    Извлекает и структурирует контент из PDF-файла, подготавливая его
    к дальнейшей обработке и добавлению в векторную базу данных.

    Args:
        file_path (Path): Путь к PDF-файлу.

    Returns:
        dict[str, Any]: Словарь с метаданными и текстом документа.
    """
    markdown_str = get_markdown_content(file_path)
    # Organize the converted document
    document = {
        "metadata": {"file_name": str(file_path), "category": "books"},
        "text": markdown_str,
    }

    return document


def split_text_into_chunks(text: str, metadata: Dict[str, Any]) -> List[Any]:
    """
    Разделяет длинный текст на более мелкие чанки, что критически важно для
    эффективной работы с LLM, так как позволяет обрабатывать большие объемы информации.

    Args:
        text (str): Текст для разделения.
        metadata (Dict[str, Any]): Метаданные для добавления к каждому чанку.

    Returns:
        List[Any]: Список чанков (документов).
    """
    start_time = time.time()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=int(os.getenv("MAX_CHUNK_SIZE")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP")),
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
        is_separator_regex=False,
    )

    chunks = text_splitter.create_documents(texts=[text], metadatas=[metadata])
    end_time = time.time()
    logger.info(f"Текст разделен на чанки за {end_time - start_time:.2f} сек")
    return chunks


def generate_chroma_db(documents) -> Optional[Chroma]:
    """
    Создает векторную базу данных ChromaDB из набора документов.
    Процесс включает векторизацию текста с помощью модели эмбеддингов
    и сохранение векторов для последующего семантического поиска.

    Args:
        documents (list): Список документов для добавления.

    Returns:
        Optional[Chroma]: Экземпляр ChromaDB или None, если нет документов.

    Raises:
        Exception: Если происходит ошибка при инициализации ChromaDB.
    """
    try:
        chroma_path = os.getenv("CHROMA_PATH")
        if CLEAR_EXISTING_DB and os.path.exists(chroma_path):
            logger.warning(f"Удаление существующей базы данных: {chroma_path}")
            shutil.rmtree(chroma_path)

        # Создаем директорию для хранения базы данных, если она не существует
        os.makedirs(chroma_path, exist_ok=True)

        if not documents:
            logger.warning("Нет документов для добавления в базу данных")
            return None

        # Инициализируем модель эмбеддингов
        embedding_start = time.time()
        embeddings = HuggingFaceEmbeddings(
            model_name=os.getenv("EMBEDDING_MODEL_NAME"),
            model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        embedding_end = time.time()
        logger.info(f"Загрузка модели эмбеддингов: {embedding_end - embedding_start:.2f} сек")

        # Подготавливаем данные для Chroma
        all_chunks = []
        for i, doc in enumerate(documents):
            chunks = split_text_into_chunks(doc["text"], doc["metadata"])
            all_chunks.extend(chunks)
            logger.info(
                f"Документ {i+1}/{len(documents)} разбит на {len(chunks)} чанков"
            )

        # Создаем векторное хранилище
        db_creation_start = time.time()
        texts = [normalize_text(chunk.page_content) for chunk in all_chunks]
        metadatas = [chunk.metadata for chunk in all_chunks]
        ids = [f"doc_{i}" for i in range(len(all_chunks))]

        chroma_db = Chroma.from_texts(
            texts=texts,
            embedding=embeddings,
            ids=ids,
            metadatas=metadatas,
            persist_directory=chroma_path,
            collection_name=os.getenv("CHROMA_COLLECTION_NAME"),
            collection_metadata={
                "hnsw:space": "cosine",
            },
        )
        db_creation_end = time.time()
        logger.success(
            f"База Chroma инициализирована и данные внесены за: {db_creation_end - db_creation_start:.2f} сек. "
            f"Добавлено {len(all_chunks)} чанков из {len(documents)} документов"
        )
        return chroma_db
    except Exception as e:
        logger.error(f"Ошибка инициализации Chroma: {e}")
        raise


def main():
    """
    Основная функция для запуска полного пайплайна:
    1. Получение и обработка документов из PDF.
    2. Создание и наполнение векторной базы данных ChromaDB.
    """
    main_start_time = time.time()
    logger.success("Старт конвертер")

    processed_document = get_documents(file_path)
    documents = [processed_document]

    generate_chroma_db(documents)

    main_end_time = time.time()
    logger.success(f"Общее время исполнения: {main_end_time - main_start_time:.2f} сек")


if __name__ == "__main__":
    main()