"""
Document ingestion: pull SEC 10-K filings, split into chunks, capture
each filing's last 1000 chars (where the directors section lives).

Returns:
    chunks            — list[Document] with company metadata attached
    director_sections — dict[str, str] mapping company name to last 1000 chars

NOTE: SEC.gov rejects requests with the default urllib User-Agent. We pass a
custom UA via header_template — that's why the loader has the extra param.
This is provided for you in the function signature; don't remove it.
"""

from typing import List, Dict, Tuple
from langchain_core.documents import Document
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_and_process_filings(
    urls: List[Tuple[str, str]],
    chunk_size: int,
    chunk_overlap: int,
    user_agent: str,
) -> Tuple[List[Document], Dict[str, str]]:
    """
    TODO M3: Build the ingestion logic.

    Goal: walk the (company, url) list, load each filing as a list of Documents,
    split them into chunks, and tag each chunk with its company name. Also
    capture the last 1000 chars of each filing into director_sections[company]
    — that's where 10-K director listings live, and the directors tool will
    use it later.

    The function signature already includes `user_agent` because SEC.gov
    blocks default urllib UAs. Make sure WebBaseLoader gets this UA.

    Print progress per company so debug runs are readable.

    See MILESTONES.md M3 for the conceptual walk-through.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    all_chunks = []
    director_sections = {}

    for company, url in urls:
        print(f"[ingestion] Loading {company} from {url} ...")

        loader = WebBaseLoader(
            url,
            header_template={"User-Agent": user_agent},
        )
        docs = loader.load()

        # Stash last 1000 chars — director listings live at the end of 10-Ks
        director_sections[company] = docs[0].page_content[-1000:]
        print(f"[ingestion] {company}: director section captured ({len(director_sections[company])} chars)")

        # Chunk the full filing
        chunks = splitter.transform_documents(docs)

        # Tag every chunk with its company name
        for chunk in chunks:
            chunk.metadata["company"] = company

        all_chunks.extend(chunks)
        print(f"[ingestion] {company}: {len(chunks)} chunks created")

    print(f"\n[ingestion] Done. Total chunks: {len(all_chunks)}, companies: {list(director_sections.keys())}")
    return all_chunks, director_sections


    raise NotImplementedError("Build load_and_process_filings — see Milestone 3")


if __name__ == "__main__":
    # Smoke test: load filings, print stats.
    from dotenv import load_dotenv
    load_dotenv()
    from config import CONFIG

    chunks, director_sections = load_and_process_filings(
        urls=CONFIG["companyFilingUrls"],
        chunk_size=CONFIG["chunkSize"],
        chunk_overlap=CONFIG["chunkOverlap"],
        user_agent=CONFIG["userAgentHeader"],
    )
    print(f"\nTotal chunks: {len(chunks)}")
    print(f"Sample chunk metadata: {chunks[0].metadata}")
    print(f"Director sections captured for: {list(director_sections.keys())}")
