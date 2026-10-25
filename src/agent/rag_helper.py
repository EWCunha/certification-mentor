from openai import OpenAI
from typing import Any, Optional, Union
from logging import getLogger
from openai.types.responses import Response
from agent.index_interface import SearchIndex


class RAGBase:
    index: SearchIndex
    llm_client: OpenAI
    model: str

    def __init__(
        self,
        index: SearchIndex,
        llm_client: OpenAI,
        model: str = "gpt-5.4-mini",
    ):
        self.index = index
        self.llm_client = llm_client
        self.model = model

    def search(
        self,
        query: str,
        num_results: int = 5,
        boost_dict: Optional[dict[str, float]] = None,
        filter_dict: Optional[dict[str, Any]] = {"enabled": True},
    ) -> Union[list, list[dict]]:
        logger = getLogger(__name__)
        logger.debug(
            f"Searching index for: {query}, boost dict: {boost_dict}, filter dict: {filter_dict}, number of results: {num_results}"
        )

        return self.index.search(
            query,
            num_results=num_results,
            boost_dict=boost_dict,
            filter_dict=filter_dict,
        )

    def llm(self, messages: list[dict[str, Any]], tools: list = []) -> Response:
        logger = getLogger(__name__)
        logger.debug(f"LLM call: {messages[-1]}")

        response = self.llm_client.responses.create(
            model=self.model, input=messages, tools=tools, max_tool_calls=6
        )

        logger.debug(f"LLM response: {response}")

        return response

    def build_context(self, search_results: list[dict]) -> str:
        logger = getLogger(__name__)
        logger.debug(f"Building context for search results: {search_results}")

        lines = []

        for doc in search_results:
            lines.append("title: " + doc["title"])
            lines.append("ccfa_domain: " + doc["ccfa_domain"])
            lines.append("notes: " + doc["notes"])
            lines.append("text: " + doc["text"])
            lines.append("")

        return "\n".join(lines).strip()
