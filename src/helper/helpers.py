from os import getenv
from config.config import CONFIG


def vec_to_str(vector: list[float]) -> str:
    return "[" + ",".join(str(x) for x in vector) + "]"


def db_connection_string() -> str:
    ingestion_config: dict[str, str] = CONFIG.get("ingestion")
    return ingestion_config["memory"]["url"].format(
        user=getenv("DATABASE_USER"),
        password=getenv("DATABASE_PASSWORD"),
        host=getenv("DATABASE_HOST"),
        port=getenv("DATABASE_PORT"),
        db_name=getenv("DATABASE_NAME"),
    )
