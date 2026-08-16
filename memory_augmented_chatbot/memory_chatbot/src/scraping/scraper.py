"""
Web scraping utilities.

Uses `requests` + `BeautifulSoup` to fetch and extract readable text from
web pages. This is the entry point of the Static Knowledge Layer's data
pipeline (Step 1 in the methodology: Web scraping -> cleaning -> chunking).

A `load_local_corpus` helper is also provided so the rest of the pipeline
(cleaning, chunking, embedding, KG extraction) can be exercised end-to-end
without live network access -- useful for demos, tests, and offline grading
environments where arbitrary outbound web requests are not available.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

USER_AGENT = (
    "Mozilla/5.0 (compatible; MemoryAugmentedChatbot/1.0; "
    "+https://github.com/example/memory-chatbot)"
)


@dataclass
class ScrapedDocument:
    source: str
    title: str
    text: str


class WebScraper:
    """Fetches and extracts clean text from a list of URLs."""

    def __init__(self, timeout: int = 10, delay: float = 0.5, max_retries: int = 2):
        self.timeout = timeout
        self.delay = delay
        self.max_retries = max_retries
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})

    def fetch(self, url: str) -> str | None:
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.session.get(url, timeout=self.timeout)
                resp.raise_for_status()
                return resp.text
            except requests.RequestException as exc:
                logger.warning("Fetch failed (%s/%s) for %s: %s", attempt, self.max_retries, url, exc)
                time.sleep(self.delay)
        return None

    @staticmethod
    def extract_text(html: str) -> tuple[str, str]:
        """Return (title, main_text) extracted from raw HTML."""
        soup = BeautifulSoup(html, "html.parser")

        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
            tag.decompose()

        title = soup.title.string.strip() if soup.title and soup.title.string else "Untitled"

        # Prefer <article> or <main> if present, else fall back to <body>.
        container = soup.find("article") or soup.find("main") or soup.body or soup
        paragraphs = [p.get_text(" ", strip=True) for p in container.find_all(["p", "li", "h1", "h2", "h3"])]
        text = "\n".join(p for p in paragraphs if p)
        return title, text

    def scrape_urls(self, urls: Iterable[str]) -> list[ScrapedDocument]:
        docs: list[ScrapedDocument] = []
        for url in urls:
            html = self.fetch(url)
            if not html:
                continue
            title, text = self.extract_text(html)
            if text.strip():
                docs.append(ScrapedDocument(source=url, title=title, text=text))
            time.sleep(self.delay)
        return docs


def load_local_corpus(corpus_dir: Path) -> list[ScrapedDocument]:
    """
    Load plain-text documents from disk, in the same `ScrapedDocument`
    shape the live scraper produces. Used as an offline stand-in for
    `WebScraper.scrape_urls` (e.g. in sandboxes without outbound web access,
    or for reproducible demos/tests).
    """
    docs: list[ScrapedDocument] = []
    for path in sorted(Path(corpus_dir).glob("*.txt")):
        raw = path.read_text(encoding="utf-8").strip()
        if not raw:
            continue
        title = path.stem.replace("_", " ").title()
        if raw.startswith("Title:"):
            first_line, _, rest = raw.partition("\n")
            title = first_line.replace("Title:", "").strip()
            raw = rest.strip()
        docs.append(ScrapedDocument(source=str(path), title=title, text=raw))
    return docs


if __name__ == "__main__":
    import sys
    from memory_augmented_chatbot.memory_chatbot.src import config

    if len(sys.argv) > 1:
        scraper = WebScraper()
        results = scraper.scrape_urls(sys.argv[1:])
    else:
        results = load_local_corpus(config.SAMPLE_CORPUS_DIR)

    for d in results:
        print(f"[{d.title}] {d.source} -> {len(d.text)} chars")
