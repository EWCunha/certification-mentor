from sqlitesearch import TextSearchIndex
from time import sleep
from logging import getLogger
from config.config import CONFIG


def add_to_persistent_memory(json_data: list[dict]):
    """Add data to the persistent memory."""
    logger = getLogger(__name__)
    logger.debug("Adding data to persistent memory")

    index = TextSearchIndex(
        text_fields=["title", "text"],
        keyword_fields=["cca_f_domain", "enabled"],
        db_path=CONFIG.get("ingestion").get("memory").get("path"),
    )

    for doc in json_data:
        index.add(doc)
        print(f"""Added: {doc["title"][:60]}...""")
        sleep(0.5)

    index.close()
