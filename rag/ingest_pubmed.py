"""
PubMed abstract ingestion via NCBI E-utilities.

Fetches up to --max abstracts matching a MeSH query (default: psychiatry[MeSH Terms]),
chunks them, embeds with sentence-transformers/all-MiniLM-L6-v2, and writes a FAISS index.

Usage:
    python -m rag.ingest_pubmed \\
        --query "psychiatry[MeSH Terms]" \\
        --max 100000 \\
        --out rag/indexes/pubmed

Rate limit: NCBI allows 10 req/s with an API key (set NCBI_API_KEY env var),
3 req/s without.  We batch in chunks of 500 records.
"""

from __future__ import annotations

import argparse
import os
import pickle
import time
from pathlib import Path

import faiss
import numpy as np
import requests
from sentence_transformers import SentenceTransformer

NCBI_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 500
CHUNK_SIZE = 512  # characters — approximate token budget for context window
CHUNK_OVERLAP = 64


def _get_pmids(query: str, max_results: int, api_key: str | None) -> list[str]:
    """Run esearch and return up to max_results PMIDs."""
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "json",
        "usehistory": "y",
    }
    if api_key:
        params["api_key"] = api_key
    resp = requests.get(f"{NCBI_BASE}/esearch.fcgi", params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return data["esearchresult"]["idlist"]


def _fetch_abstracts_batch(pmids: list[str], api_key: str | None) -> list[dict]:
    """Fetch abstracts for a batch of PMIDs via efetch."""
    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "rettype": "abstract",
        "retmode": "json",
    }
    if api_key:
        params["api_key"] = api_key

    resp = requests.get(f"{NCBI_BASE}/efetch.fcgi", params=params, timeout=60)
    resp.raise_for_status()

    records = []
    try:
        articles = resp.json().get("PubmedArticleSet", {}).get("PubmedArticle", [])
        if isinstance(articles, dict):
            articles = [articles]
        for art in articles:
            medline = art.get("MedlineCitation", {})
            article = medline.get("Article", {})
            pmid = str(medline.get("PMID", {}).get("#text", ""))
            title = article.get("ArticleTitle", "")
            if isinstance(title, dict):
                title = title.get("#text", "")
            abstract_obj = article.get("Abstract", {}).get("AbstractText", "")
            if isinstance(abstract_obj, list):
                abstract = " ".join(
                    (t.get("#text", "") if isinstance(t, dict) else str(t))
                    for t in abstract_obj
                )
            elif isinstance(abstract_obj, dict):
                abstract = abstract_obj.get("#text", "")
            else:
                abstract = str(abstract_obj)
            if abstract and len(abstract) > 50:
                records.append({"pmid": pmid, "title": title, "abstract": abstract})
    except Exception:
        pass
    return records


def _chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap
    return chunks


def ingest(
    query: str,
    max_results: int,
    out_dir: Path,
    api_key: str | None = None,
    embed_batch: int = 256,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Searching PubMed: {query!r} (max {max_results})")
    pmids = _get_pmids(query, max_results, api_key)
    print(f"Found {len(pmids)} PMIDs — fetching abstracts in batches of {BATCH_SIZE} ...")

    documents: list[dict] = []
    for i in range(0, len(pmids), BATCH_SIZE):
        batch = pmids[i: i + BATCH_SIZE]
        records = _fetch_abstracts_batch(batch, api_key)
        documents.extend(records)
        delay = 0.11 if api_key else 0.35
        time.sleep(delay)
        if i % 5000 == 0 and i > 0:
            print(f"  {i}/{len(pmids)} fetched, {len(documents)} with abstracts so far")

    print(f"Fetched {len(documents)} abstracts.  Chunking ...")
    chunks: list[str] = []
    metadata: list[dict] = []
    for doc in documents:
        full_text = f"{doc['title']}\n\n{doc['abstract']}"
        for chunk in _chunk_text(full_text):
            chunks.append(chunk)
            metadata.append({"pmid": doc["pmid"], "title": doc["title"]})

    print(f"Total chunks: {len(chunks)}.  Embedding with {EMBED_MODEL} ...")
    encoder = SentenceTransformer(EMBED_MODEL)
    embeddings = encoder.encode(
        chunks, batch_size=embed_batch, show_progress_bar=True, normalize_embeddings=True
    )
    embeddings = np.array(embeddings, dtype=np.float32)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner product on L2-normalised vectors = cosine sim.
    index.add(embeddings)

    faiss.write_index(index, str(out_dir / "faiss.index"))
    with open(out_dir / "chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)
    with open(out_dir / "metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)

    print(f"\nIndex written to {out_dir}:")
    print(f"  faiss.index  ({index.ntotal} vectors, dim={dim})")
    print(f"  chunks.pkl   ({len(chunks)} text chunks)")
    print(f"  metadata.pkl ({len(metadata)} records with PMID + title)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", default="psychiatry[MeSH Terms] OR mental health[MeSH Terms]")
    parser.add_argument("--max", dest="max_results", type=int, default=100_000)
    parser.add_argument("--out", type=Path, default=Path("rag/indexes/pubmed"))
    parser.add_argument("--embed-batch", type=int, default=256)
    args = parser.parse_args()

    api_key = os.environ.get("NCBI_API_KEY")
    if not api_key:
        print("Tip: set NCBI_API_KEY env var for 10 req/s (vs 3 req/s without key)")

    ingest(
        query=args.query,
        max_results=args.max_results,
        out_dir=args.out,
        api_key=api_key,
        embed_batch=args.embed_batch,
    )


if __name__ == "__main__":
    main()
