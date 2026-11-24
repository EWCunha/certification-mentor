from dotenv import load_dotenv
from logging import getLogger, basicConfig
from sqlitesearch import TextSearchIndex
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from psycopg import connect
from toyaikit.tools import Tools
from minsearch import Index
from agent.agent import Agent
from agent.rag_helper import RAGBase
from agent.rag_vector import RAGPgVector
from state.state import State
from ingestion.ingestion import ingest
from monitoring.database import create_database
from config.config import CONFIG
from helper.helpers import db_connection_string, parse_log_level


load_dotenv()


def main():
    # logging
    basicConfig(level=parse_log_level(CONFIG.get("logging").get("level")))
    logger = getLogger(__name__)
    logger.debug("Starting certification mentor")

    # ingestion
    ingest()

    # monitoring
    logger.debug("Creating monitoring database")
    logger.debug(create_database())

    # application
    if CONFIG.get("ingestion").get("memory").get("type") == "runtime":
        logger.debug("Initializing index")
        index = Index(
            text_fields=["title", "text"],
            keyword_fields=["cca_f_domain", "enabled"],
        )

        logger.debug("Initializing RAG")
        rag = RAGBase(
            index=index,
            llm_client=OpenAI(),
        )

    elif CONFIG.get("ingestion").get("memory").get("type") == "persistent":
        logger.debug("Initializing index")
        index = TextSearchIndex(
            text_fields=["title", "text"],
            keyword_fields=["cca_f_domain", "enabled"],
            db_path=CONFIG.get("ingestion").get("memory").get("path"),
        )

        logger.debug("Initializing RAG")
        rag = RAGBase(
            index=index,
            llm_client=OpenAI(),
        )
    elif CONFIG.get("ingestion").get("memory").get("type") == "vector_db":
        logger.debug("Initializing RAG")
        embedder = SentenceTransformer(
            CONFIG.get("ingestion").get("memory").get("embedding_model")
        )

        logger.debug("Connecting to database")
        conn = connect(db_connection_string())

        logger.debug("Initializing RAG")
        rag = RAGPgVector(
            embedder=embedder,
            conn=conn,
            llm_client=OpenAI(),
        )
    else:
        raise ValueError(
            f"Invalid memory type: {CONFIG.get('ingestion').get('memory').get('type')}"
        )

    logger.debug("Initializing state")
    state = State(rag)

    logger.debug("Initializing tools")
    tools = Tools()

    logger.debug("Adding tools to tools")
    tools.add_tools(state)

    logger.debug("Initializing agent")
    agent = Agent(rag, tools=tools)

    agent.loop(state)
