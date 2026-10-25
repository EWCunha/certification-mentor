from json import load
from sqlitesearch import TextSearchIndex
from time import sleep
from ingestion.crawler import Crawler
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

        self.sources = self._fetch_all(self.sources)
        if self.ingestion_config["memory"]["type"] == "persistent":
            self._add_to_persistent_memory(self.sources)

    def _fetch_all(self, sources: list[dict]) -> list[dict]:
        """Fetch all the URLs in the sources file."""

        for source in sources:
            try:
                page = self.crawler.crawl(
                    source["url"],
                )
                page_data = page.json()
                source["title"] = page_data["title"]
                source["text"] = page_data["text"]
            except Exception as e:
                print(f"Error crawling {source}: {e}")

        return sources

    def _add_to_persistent_memory(self, json_data: list[dict]):
        """Add data to the persistent memory."""

        index = TextSearchIndex(
            text_fields=["title", "text"],
            keyword_fields=["cca_f_domain", "enabled"],
            db_path=self.ingestion_config["memory"]["path"],
        )

        for doc in json_data:
            index.add(doc)
            print(f"""Added: {doc["title"][:60]}...""")
            sleep(0.5)

        index.close()
