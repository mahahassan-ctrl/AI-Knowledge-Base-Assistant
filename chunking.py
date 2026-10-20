
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)

# Folder containing the three PDF files
DATA_PATH = "/workspaces/AI-Knowledge-Base-Assistant/data"

# All three knowledge-base documents
DEFAULT_PDF_FILES = [
    "employee-handbook-01.pdf",
    "Enterprise HR and IT Policy Document.pdf",
    "security-policy-03.pdf",
]


def load_all_pdfs(data_path=DATA_PATH, filenames=None):
    """
    Load PDF pages as LangChain Documents.
    Each page retains metadata such as source and page number.
    """
    if filenames is None:
        filenames = DEFAULT_PDF_FILES

    documents = []

    for filename in filenames:
        file_path = os.path.join(data_path, filename)

        if not os.path.isfile(file_path):
            raise FileNotFoundError(
                f"PDF not found: {file_path}\n"
                "Check the filename and data folder."
            )

        print(f"Loading: {filename}")

        loader = PyPDFLoader(file_path)
        pages = loader.load()

        # Keep the source filename and page number in metadata.
        for page in pages:
            page.metadata["source"] = filename

        documents.extend(pages)
        print(f"  Pages loaded: {len(pages)}")

    if not documents:
        raise ValueError("No PDF pages were loaded.")

    return documents


def split_documents(
    documents,
    strategy="recursive",
    chunk_size=500,
    chunk_overlap=100,
):
    """
    Split documents using one of three strategies:
    1. recursive: split using paragraphs, lines, words, then characters.
    2. fixed: split into fixed-size character windows.
    3. structure: prefer paragraph and sentence boundaries.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be non-negative and smaller than chunk_size."
        )

    if strategy == "recursive":
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True,
        )

    elif strategy == "fixed":
        # Fixed-size character windows with explicit overlap.
        splitter = CharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    elif strategy == "structure":
        splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ". ", " ", ""],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True,
        )

    else:
        raise ValueError(
            "Unknown strategy. Choose 'recursive', 'fixed', or 'structure'."
        )

    chunks = splitter.split_documents(documents)

    # Record the strategy so evaluation results can be compared later.
    for chunk in chunks:
        chunk.metadata["chunking_strategy"] = strategy

    return chunks


if __name__ == "__main__":
    # Load all three PDFs once.
    docs = load_all_pdfs()

    print(f"\nTotal PDF pages loaded: {len(docs)}")

    # Compare all three chunking strategies.
    strategies = ["recursive", "fixed", "structure"]

    for strategy in strategies:
        chunks = split_documents(
            documents=docs,
            strategy=strategy,
            chunk_size=500,
            chunk_overlap=100,
        )

        print(f"\n{'=' * 50}")
        print(f"Strategy: {strategy}")
        print(f"Total chunks: {len(chunks)}")

        if chunks:
            average_length = sum(
                len(chunk.page_content) for chunk in chunks
            ) / len(chunks)

            print(f"Average chunk length: {average_length:.1f} characters")
            print("\nSample chunk:")
            print(chunks[0].page_content[:500])

            print("\nSample metadata:")
            print(chunks[0].metadata)

    print("\nChunking comparison completed.")
