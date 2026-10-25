from sqlitesearch import TextSearchIndex
from openai import OpenAI
from dotenv import load_dotenv
from toyaikit.tools import Tools
from logging import getLogger, basicConfig
from agent.agent import Agent
from state.state import State
from agent.rag_helper import RAGBase
from config.config import CONFIG

load_dotenv()


def main():
    basicConfig(level=CONFIG.get("logging").get("level"))
    logger = getLogger(__name__)
    logger.debug("Starting certification mentor")

    logger.debug("Initializing index")
    index = TextSearchIndex(
        text_fields=["title", "text"],
        keyword_fields=["cca_f_domain", "enabled"],
        db_path=CONFIG.get("ingestion").get("memory").get("path"),
    )

    logger.debug("Initializing state")
    state = State(index)

    logger.debug("Initializing tools")
    tools = Tools()

    logger.debug("Adding tools to tools")
    tools.add_tools(state)

    logger.debug("Initializing RAG")
    rag = RAGBase(
        index=index,
        llm_client=OpenAI(),
    )

    logger.debug("Initializing agent")
    agent = Agent(rag, tools=tools)

    agent.loop(state)
