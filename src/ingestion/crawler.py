from httpx import Client, Response
from bs4 import BeautifulSoup
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse

_REMOVE_TAGS = ["script", "style", "header", "footer", "nav", "aside"]


class CrawlError(Exception):
    """Exception raised when a crawl operation fails."""

    pass


@dataclass
class CrawledPage:
    title: Optional[str]
    text: str
    url: str

    def json(self) -> dict:
        return {
            "title": self.title,
            "text": self.text,
            "url": self.url,
        }


class Crawler:
    client: Client
    timeout: int

    def __init__(self, timeout: int = 30):
        self.client = Client(timeout=timeout)
        self.timeout = timeout

    def crawl(self, url: str) -> CrawledPage:
        """Crawl a URL and return the crawled page."""

        try:
            response = self._fetch_url(url)
        except Exception as e:
            raise CrawlError(f"Failed to fetch URL: {url}. Error: {e}")

        return self._extract_info(response, url)

    def _fetch_url(self, url: str) -> Response:
        """Fetch the content of a URL."""

        response = self.client.get(url, timeout=self.timeout, follow_redirects=True)
        response.raise_for_status()
        return response

    def _extract_info(self, response: Response, url: str) -> CrawledPage:
        """Extract the information from an HTML/Markdown document."""

        if self._is_markdown(url, response.headers.get("content-type")):
            text = response.text.strip()
            title = None
            for line in text.splitlines():
                line = line.strip()
                if line.startswith("# "):
                    title = line.removeprefix("#").strip()
                    break

            if title is None:
                title = urlparse(url).path
                title = title.split("/")[-1]
                title = title.removesuffix(".md").removesuffix(".markdown")
                title = title.replace("-", " ")
                title = title.title()
        else:
            soup = BeautifulSoup(response.text, "html.parser")

            if soup.title is None:
                title = None
            else:
                title = soup.title.get_text(strip=True)
                if title == "":
                    title = None

            elements = soup.find_all(_REMOVE_TAGS)
            for element in elements:
                element.decompose()

            text = soup.get_text(separator="\n", strip=True)

        return CrawledPage(title=title, text=text, url=url)

    def _path_ends_with_markdown(self, path: str) -> bool:
        lowered = path.lower()
        return lowered.endswith(".md") or lowered.endswith(".markdown")

    def _is_markdown_url(self, url: str) -> bool:
        """True when the URL path looks like a markdown resource."""

        return self._path_ends_with_markdown(urlparse(url).path)

    def _is_markdown(self, url: str, content_type: Optional[str] = None) -> bool:
        """True when the response should be treated as markdown."""

        if self._is_markdown_url(url):
            return True

        if not content_type:
            return False

        # Strip charset, e.g. "text/markdown; charset=utf-8"
        media_type = content_type.split(";", 1)[0].strip().lower()
        return media_type in {"text/markdown", "text/x-markdown"}
