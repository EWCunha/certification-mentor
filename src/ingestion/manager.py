from json import load
from logging import getLogger
from sqlitesearch import TextSearchIndex
from time import sleep
from sentence_transformers import SentenceTransformer
from psycopg import connect
from tqdm.auto import tqdm
from ingestion.crawler import Crawler
from helper.helpers import vec_to_str, db_connection_string
from config.config import CONFIG


class IngestionManager:
    ingestion_config: dict
    sources: list[dict]
    crawler: Crawler

    def __init__(self):
        self.ingestion_config = CONFIG.get("ingestion")
        with open(CONFIG.get("ingestion").get("sources_path"), "r") as f:
            self.sources = load(f)

        self.crawler = Crawler(timeout=self.ingestion_config["timeout"])

    def run(self):
        if not self.ingestion_config["enabled"]:
            return

        logger = getLogger(__name__)
        logger.info("Ingesting data...")
        self.sources = self._fetch_all(self.sources)
        if self.ingestion_config["memory"]["type"] == "persistent":
            logger.info("Adding data to persistent memory...")
            self._add_to_persistent_memory(self.sources)
        elif self.ingestion_config["memory"]["type"] == "vector_db":
            logger.info("Embedding data...")
            vectors = self._embed_data(self.sources)
            logger.info("Adding data to vector database...")
            self._add_to_vector_database(self.sources, vectors)

    def _fetch_all(self, sources: list[dict]) -> list[dict]:
        """Fetch all the URLs in the sources file."""

        logger = getLogger(__name__)
        logger.info(f"Fetching {len(sources)} sources...")
        for source in sources:
            try:
                page = self.crawler.crawl(
                    source["url"],
                )
                page_data = page.json()
                source["title"] = page_data["title"]
                source["text"] = page_data["text"]
            except Exception as e:
                logger.error(f"Error crawling {source}: {e}")

        return sources

    def _add_to_persistent_memory(self, json_data: list[dict]):
        """Add data to the persistent memory."""

        logger = getLogger(__name__)
        logger.info(f"Adding {len(json_data)} documents to persistent memory...")

        index = TextSearchIndex(
            text_fields=["title", "text"],
            keyword_fields=["cca_f_domain", "enabled"],
            db_path=self.ingestion_config["memory"]["path"],
        )

        for doc in json_data:
            index.add(doc)
            logger.info(f"""Added: {doc["title"][:60]}...""")
            sleep(0.5)

        index.close()

    def _embed_data(self, json_data: list[dict]) -> list[list[float]]:
        """Embed the data using the embedding model."""

        logger = getLogger(__name__)
        logger.info(f"Embedding {len(json_data)} documents...")
        model = SentenceTransformer(self.ingestion_config["memory"]["embedding_model"])
        texts = [doc["title"] + " " + doc["text"] for doc in json_data]

        batch_size = 50
        vectors = []

        for i in tqdm(range(0, len(texts), batch_size)):
            batch = texts[i : i + batch_size]
            batch_vectors = model.encode(batch)
            vectors.extend(batch_vectors)

        return vectors

    def _add_to_vector_database(
        self, json_data: list[dict], vectors: list[list[float]]
    ):
        logger = getLogger(__name__)
        logger.info(f"Adding {len(json_data)} documents to vector database...")

        connection_string = db_connection_string()
        conn = connect(connection_string)
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

        conn.execute(
            """
            DROP TABLE IF EXISTS documents
        """
        )

        conn.execute(
            """
            CREATE TABLE documents (
                id SERIAL PRIMARY KEY,
                source_id TEXT,
                title TEXT,
                url TEXT,
                cca_f_domain TEXT,
                text TEXT,
                embedding vector(384),
                enabled BOOLEAN
            )
            """
        )

        for doc, vec in tqdm(zip(json_data, vectors), total=len(json_data)):
            conn.execute(
                """
                INSERT INTO documents (source_id, title, url, cca_f_domain, text, embedding, enabled)
                VALUES (%s, %s, %s, %s, %s, %s::vector, %s)
                """,
                (
                    doc["id"],
                    doc["title"],
                    doc["url"],
                    doc["cca_f_domain"],
                    doc["text"],
                    vec_to_str(vec),
                    doc["enabled"],
                ),
            )

        conn.commit()
