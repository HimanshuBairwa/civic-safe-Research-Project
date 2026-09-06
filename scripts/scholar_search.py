#!/usr/bin/env python3
"""
Scholar Search Utility for CIVIC-SAFE
=====================================
Direct scholarly literature search via public APIs (arXiv & Crossref).
Requires no API keys, works without browser/gateway tools, and outputs
clean paper metadata and BibTeX entries ready for references.bib.

Usage:
    python scripts/scholar_search.py "performative prediction conformal"
    python scripts/scholar_search.py "spatiotemporal crime forecasting" --limit 5
    python scripts/scholar_search.py "Perdomo 2020 performative" --bibtex
"""

import argparse
import json
import re
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Any


def clean_text(text: str) -> str:
    """Clean whitespace and newlines from strings."""
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", "", text)
    return " ".join(text.split())


def search_arxiv(query: str, limit: int = 5) -> list[dict[str, Any]]:
    """Search arXiv API for preprints matching query."""
    encoded_query = urllib.parse.quote(query)
    url = f"http://export.arxiv.org/api/query?search_query={encoded_query}&max_results={limit}&sortBy=relevance"
    req = urllib.request.Request(url, headers={"User-Agent": "CIVIC-SAFE-Research/1.0"})

    papers = []
    try:
        with urllib.request.urlopen(req, timeout=12) as resp:
            data = resp.read()
        tree = ET.fromstring(data)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        for entry in tree.findall("atom:entry", ns):
            title = clean_text(entry.findtext("atom:title", default="", namespaces=ns))
            summary = clean_text(entry.findtext("atom:summary", default="", namespaces=ns))
            published = clean_text(entry.findtext("atom:published", default="", namespaces=ns))
            year = published[:4] if len(published) >= 4 else "recent"
            entry_id = clean_text(entry.findtext("atom:id", default="", namespaces=ns))
            arxiv_id = entry_id.split("/abs/")[-1] if "/abs/" in entry_id else entry_id

            authors = []
            for author in entry.findall("atom:author", ns):
                name = author.findtext("atom:name", default="", namespaces=ns)
                if name:
                    authors.append(clean_text(name))

            first_author_last = authors[0].split()[-1].lower() if authors else "anon"
            cite_key = f"{first_author_last}{year}{re.sub(r'[^a-zA-Z0-9]', '', title.split()[0].lower())}"

            bibtex = (
                f"@article{{{cite_key},\n"
                f"  title={{{title}}},\n"
                f"  author={{{' and '.join(authors)}}},\n"
                f"  journal={{arXiv preprint arXiv:{arxiv_id}}},\n"
                f"  year={{{year}}},\n"
                f"  url={{{entry_id}}}\n"
                f"}}"
            )

            papers.append({
                "source": "arXiv",
                "title": title,
                "authors": authors,
                "year": year,
                "id": arxiv_id,
                "url": entry_id,
                "abstract": summary,
                "cite_key": cite_key,
                "bibtex": bibtex,
            })
    except Exception as e:
        sys.stderr.write(f"[arXiv search warning: {e}]\n")
    return papers


def search_crossref(query: str, limit: int = 5) -> list[dict[str, Any]]:
    """Search Crossref API for peer-reviewed journal & conference papers."""
    encoded_query = urllib.parse.quote(query)
    url = f"https://api.crossref.org/works?query={encoded_query}&rows={limit}"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "CIVIC-SAFE-Research/1.0 (mailto:bairwahimanshu29@gmail.com)"}
    )

    papers = []
    try:
        with urllib.request.urlopen(req, timeout=12) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        items = data.get("message", {}).get("items", [])
        for item in items:
            title_list = item.get("title", [])
            title = clean_text(title_list[0]) if title_list else "Untitled"

            authors = []
            for a in item.get("author", []):
                given = a.get("given", "")
                family = a.get("family", "")
                if family and given:
                    authors.append(f"{given} {family}")
                elif family:
                    authors.append(family)

            date_parts = (
                item.get("published-print", {}).get("date-parts")
                or item.get("published-online", {}).get("date-parts")
                or item.get("created", {}).get("date-parts")
                or [[None]]
            )
            year = str(date_parts[0][0]) if date_parts and date_parts[0] and date_parts[0][0] else "recent"

            doi = item.get("DOI", "")
            url = item.get("URL", f"https://doi.org/{doi}" if doi else "")
            venue_list = item.get("container-title", [])
            venue = clean_text(venue_list[0]) if venue_list else "Proceedings/Journal"
            abstract = clean_text(item.get("abstract", ""))

            first_author_last = authors[0].split()[-1].lower() if authors else "anon"
            cite_key = f"{first_author_last}{year}{re.sub(r'[^a-zA-Z0-9]', '', title.split()[0].lower())}"

            bibtex = (
                f"@article{{{cite_key},\n"
                f"  title={{{title}}},\n"
                f"  author={{{' and '.join(authors)}}},\n"
                f"  journal={{{venue}}},\n"
                f"  year={{{year}}},\n"
                f"  doi={{{doi}}}\n"
                f"}}"
            )

            papers.append({
                "source": "Crossref",
                "title": title,
                "authors": authors,
                "year": year,
                "id": doi,
                "url": url,
                "abstract": abstract,
                "venue": venue,
                "cite_key": cite_key,
                "bibtex": bibtex,
            })
    except Exception as e:
        sys.stderr.write(f"[Crossref search warning: {e}]\n")
    return papers


def main():
    parser = argparse.ArgumentParser(description="Search scholarly literature across arXiv and Crossref")
    parser.add_argument("query", type=str, help="Search query (e.g. 'conformal prediction performative')")
    parser.add_argument("--source", choices=["all", "arxiv", "crossref"], default="all", help="Target API source")
    parser.add_argument("--limit", type=int, default=5, help="Max results per source")
    parser.add_argument("--bibtex", action="store_true", help="Print BibTeX entries for matching papers")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")

    args = parser.parse_args()

    results = []
    if args.source in ("all", "arxiv"):
        results.extend(search_arxiv(args.query, limit=args.limit))
    if args.source in ("all", "crossref"):
        results.extend(search_crossref(args.query, limit=args.limit))

    if args.json:
        print(json.dumps(results, indent=2))
        return

    if not results:
        print(f"No results found for query: '{args.query}'")
        return

    print(f"\nFound {len(results)} paper(s) for '{args.query}':\n" + "=" * 70)
    for i, p in enumerate(results, 1):
        author_str = ", ".join(p["authors"][:3])
        if len(p["authors"]) > 3:
            author_str += " et al."
        print(f"[{i}] [{p['source']} ({p['year']})] {p['title']}")
        print(f"    Authors: {author_str or 'N/A'}")
        if p.get("venue"):
            print(f"    Venue:   {p['venue']}")
        print(f"    Link:    {p['url']}")
        if p.get("abstract"):
            snippet = p["abstract"][:240] + "..." if len(p["abstract"]) > 240 else p["abstract"]
            print(f"    Summary: {snippet}")
        if args.bibtex:
            print("\n    BibTeX:")
            for line in p["bibtex"].split("\n"):
                print(f"      {line}")
        print("-" * 70)


if __name__ == "__main__":
    main()
