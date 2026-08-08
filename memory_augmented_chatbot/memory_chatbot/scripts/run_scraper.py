"""
Run the `WebScraper` against URLs listed in `data/source_urls.txt`.

Saves results to a JSONL file (one JSON object per line) with keys:
`source`, `title`, `text`.

Usage:
    python scripts/run_scraper.py --urls data/source_urls.txt --out artifacts/scraped.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

# Ensure project root is on sys.path so `src` package can be imported
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.scraping.scraper import WebScraper, ScrapedDocument


def read_urls(path: Path) -> list[str]:
    raw = path.read_text(encoding="utf-8")
    lines = [ln.strip() for ln in raw.splitlines()]
    urls = [ln for ln in lines if ln and not ln.startswith("#")]
    return urls


def write_jsonl(docs: Iterable[ScrapedDocument], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for d in docs:
            json.dump({"source": d.source, "title": d.title, "text": d.text}, fh, ensure_ascii=False)
            fh.write("\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--urls", type=Path, default=Path("data/source_urls.txt"))
    p.add_argument("--out", type=Path, default=Path("artifacts/scraped.jsonl"))
    args = p.parse_args()

    urls = read_urls(args.urls)
    if not urls:
        print(f"No URLs found in {args.urls}")
        return

    scraper = WebScraper()
    print(f"Scraping {len(urls)} URLs...")
    docs = scraper.scrape_urls(urls)
    print(f"Scraped {len(docs)} documents; saving to {args.out}")
    write_jsonl(docs, args.out)


if __name__ == "__main__":
    main()
