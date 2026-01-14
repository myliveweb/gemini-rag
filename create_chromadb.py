import time
from langchain_huggingface import HuggingFaceEmbeddings
import torch
from loguru import logger
from langchain_chroma import Chroma
import json

CHROMA_PATH = "./shop_chroma_db"
COLLECTION_NAME = "shop_data"

# Список данных о товарах
SHOP_DATA = [
    {
        "text": 'Игровой ноутбук ASUS ROG Strix G16: процессор Intel Core i9-13980HX, видеокарта NVIDIA RTX 4070 8ГБ, 32 ГБ RAM DDR5, SSD 1 ТБ, экран 16" QHD+ 240Гц, цена 185000 руб.',
        "metadata": {
            "id": "1",
            "type": "product",
            "category": "laptops",
            "price": 185000,
            "stock": 4,
        },
    },
    {
        "text": 'Смартфон Apple iPhone 15 Pro Max: 256 ГБ, титановый корпус, чип A17 Pro, основная камера 48 Мп с 5-кратным оптическим зумом, экран Super Retina XDR 6.7", цена 142000 руб.',
        "metadata": {
            "id": "2",
            "type": "product",
            "category": "smartphones",
            "price": 142000,
            "stock": 7,
        },
    },
    {
        "text": 'Профессиональный монитор Dell UltraSharp U2723QE: 27 дюймов, разрешение 4K UHD, матрица IPS Black для глубокого черного, охват 98% DCI-P3, USB-C Hub с зарядкой 90W, цена 68000 руб.',
        "metadata": {
            "id": "3",
            "type": "product",
            "category": "monitors",
            "price": 68000,
            "stock": 2,
        },
    },
    {
        "text": 'Беспроводные наушники Bose QuietComfort Ultra: система активного шумоподавления (ANC), поддержка пространственного аудио, Bluetooth 5.3, до 24 часов работы, цена 45000 руб.',
        "metadata": {
            "id": "4",
            "type": "product",
            "category": "audio",
            "price": 45000,
            "stock": 5,
        },
    },
    {
        "text": 'Кофемашина автоматическая Jura E8 Piano Black: 17 программ приготовления, встроенная кофемолка P.A.G.2, интеллектуальная система очистки воды, сенсорный дисплей, цена 138000 руб.',
        "metadata": {
            "id": "5",
            "type": "product",
            "category": "home_appliances",
            "price": 138000,
            "stock": 1,
        },
    },
    {
        "text": 'Электронная книга Onyx Boox Note Air 3 C: цветной экран E-Ink Kaleido 3 (10.3"), восьмиядерный процессор, Android 12, поддержка стилуса и распознавания текста, цена 59900 руб.',
        "metadata": {
            "id": "6",
            "type": "product",
            "category": "e-readers",
            "price": 59900,
            "stock": 3,
        },
    },
    {
        "text": 'Видеокарта MSI GeForce RTX 4080 Super 16GB Gaming X Trio: архитектура Ada Lovelace, техпроцесс 4нм, поддержка DLSS 3.0 и трассировки лучей в реальном времени, цена 125000 руб.',
        "metadata": {
            "id": "7",
            "type": "product",
            "category": "pc_components",
            "price": 125000,
            "stock": 0,
        },
    },
    {
        "text": 'Умная колонка Яндекс Станция Макс: мощность звука 65 Вт, поддержка видео 4K, встроенный хаб управления Zigbee, голосовой помощник Алиса и LED-экран, цена 31990 руб.',
        "metadata": {
            "id": "8",
            "type": "product",
            "category": "smart_home",
            "price": 31990,
            "stock": 9,
        },
    },
    {
        "text": 'Беззеркальный фотоаппарат Sony Alpha A7 IV Body: полнокадровый сенсор 33 Мп, запись видео 4K 60p 10-bit, продвинутый автофокус по глазам людей и животных, цена 210000 руб.',
        "metadata": {
            "id": "9",
            "type": "product",
            "category": "cameras",
            "price": 210000,
            "stock": 2,
        },
    },
    {
        "text": 'Роутер ASUS RT-AX88U Pro: стандарт Wi-Fi 6 (802.11ax), скорость до 6000 Мбит/с, двухдиапазонный, 4 порта 2.5G WAN/LAN, поддержка технологии AiMesh, цена 27500 руб.',
        "metadata": {
            "id": "10",
            "type": "product",
            "category": "networking",
            "price": 27500,
            "stock": 6,
        },
    },
]


def generate_chroma_db():
    """
    Создает и сохраняет векторную базу данных ChromaDB из структурированных
    данных о товарах (SHOP_DATA). Эта база данных служит базой знаний
    для LLM, позволяя выполнять семантический поиск по каталогу товаров.

    Returns:
        Chroma: Экземпляр созданной базы данных ChromaDB.

    Raises:
        Exception: Если возникает ошибка при создании базы данных.
    """
    try:
        start_time = time.time()

        logger.info("Загрузка модели эмбеддингов...")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        logger.info(f"Модель загружена за {time.time() - start_time:.2f} сек")

        logger.info("Создание Chroma DB...")
        chroma_db = Chroma.from_texts(
            texts=[item["text"] for item in SHOP_DATA],
            embedding=embeddings,
            ids=[str(item["metadata"]["id"]) for item in SHOP_DATA],
            metadatas=[item["metadata"] for item in SHOP_DATA],
            persist_directory=CHROMA_PATH,
            collection_name=COLLECTION_NAME,
        )
        logger.info(f"Chroma DB создана за {time.time() - start_time:.2f} сек")

        return chroma_db
    except Exception as e:
        logger.error(f"Ошибка: {e}")
        raise


# Вывод результата
if __name__ == "__main__":
    generate_chroma_db()