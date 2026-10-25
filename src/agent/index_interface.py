from typing import Protocol, Any, Optional


class SearchIndex(Protocol):
    def search(
        self,
        query: str,
        *,
        num_results: int = 5,
        boost_dict: Optional[dict[str, float]] = None,
    ) -> list[dict[str, Any]]: ...
