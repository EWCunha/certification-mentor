from ingestion.manager import IngestionManager
from config.config import CONFIG


def ingest():
    ingestion_manager = IngestionManager()
    ingestion_manager.run()

    CONFIG.conf["ingestion"]["enabled"] = False
    CONFIG.save()


if __name__ == "__main__":
    ingest()
