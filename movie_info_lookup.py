from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen


USER_AGENT = "long2shorts/0.1 (movie-info lookup)"


def fetch_json(url: str) -> Any:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=15) as response:
        return json.loads(response.read().decode("utf-8"))


def search_titles(query: str, lang: str) -> list[dict]:
    params = urlencode(
        {
            "action": "opensearch",
            "search": query,
            "limit": 5,
            "namespace": 0,
            "format": "json",
        }
    )
    data = fetch_json(f"https://{lang}.wikipedia.org/w/api.php?{params}")
    titles = data[1] if len(data) > 1 else []
    descriptions = data[2] if len(data) > 2 else []
    urls = data[3] if len(data) > 3 else []
    results = []
    for index, title in enumerate(titles):
        results.append(
            {
                "title": title,
                "description": descriptions[index] if index < len(descriptions) else "",
                "url": urls[index] if index < len(urls) else "",
            }
        )
    return results


def fetch_page_summary(title: str, lang: str) -> dict:
    params = urlencode(
        {
            "action": "query",
            "redirects": 1,
            "prop": "extracts|info",
            "inprop": "url",
            "exintro": 1,
            "explaintext": 1,
            "titles": title,
            "format": "json",
        }
    )
    data = fetch_json(f"https://{lang}.wikipedia.org/w/api.php?{params}")
    pages = data.get("query", {}).get("pages", {})
    for page in pages.values():
        if int(page.get("pageid", -1)) < 0:
            continue
        return {
            "title": page.get("title", title),
            "summary": (page.get("extract") or "").strip(),
            "url": page.get("fullurl", ""),
        }
    return {"title": title, "summary": "", "url": ""}


def choose_best_result(query: str, results: list[dict]) -> Optional[dict]:
    if not results:
        return None
    lowered = query.strip().lower()
    media_keywords = ["영화", "드라마", "텔레비전", "시리즈", "film", "television", "tv"]
    for result in results:
        title_lower = result["title"].strip().lower()
        if any(keyword in title_lower for keyword in ["(영화)", "(드라마)", "(텔레비전 시리즈)", "(film)", "(tv series)"]):
            if lowered in title_lower:
                return result
    for result in results:
        desc_lower = (result.get("description") or "").strip().lower()
        if any(keyword in desc_lower for keyword in media_keywords):
            if lowered in result["title"].strip().lower():
                return result
    for result in results:
        if result["title"].strip().lower() == lowered:
            return result
    for result in results:
        if lowered in result["title"].strip().lower():
            return result
    return results[0]


def lookup_movie_info(query: str) -> dict:
    query = query.strip()
    if not query:
        raise ValueError("query must not be empty")

    for lang in ["ko", "en"]:
        results = search_titles(query, lang)
        best = choose_best_result(query, results)
        if not best:
            continue
        summary = fetch_page_summary(best["title"], lang)
        return {
            "query": query,
            "matched_title": summary["title"] or best["title"],
            "language": lang,
            "description": best.get("description", ""),
            "summary": summary.get("summary", ""),
            "url": summary.get("url") or best.get("url", ""),
            "candidates": results,
        }

    return {
        "query": query,
        "matched_title": "",
        "language": "",
        "description": "",
        "summary": "",
        "url": "",
        "candidates": [],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Look up basic movie info from Wikipedia.")
    parser.add_argument("--query", required=True, help="Movie or drama title to search.")
    parser.add_argument("--output", type=Path, default=None, help="Optional output JSON path.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = lookup_movie_info(args.query)
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
        print(f"saved={args.output.resolve()}")
    else:
        print(payload)


if __name__ == "__main__":
    main()
