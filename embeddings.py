
import os
import re

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from chunking import load_all_pdfs, split_documents


# Folder where separate vector stores will be saved
DB_PATH = "vectorstore"

# Two embedding models to compare
EMBEDDING_MODELS = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "mpnet": "sentence-transformers/all-mpnet-base-v2",
}


def create_vector_store(chunks, model_name, model_key):
    """
    Create and save a FAISS vector store using the selected
    embedding model.
    """
    print(f"\nLoading embedding model: {model_name}")

    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        encode_kwargs={"normalize_embeddings": True},
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Store each model in its own directory
    model_folder = os.path.join(DB_PATH, model_key)
    os.makedirs(model_folder, exist_ok=True)

    vectorstore.save_local(model_folder)

    print(f"Vector store saved to: {model_folder}")

    return vectorstore


if __name__ == "__main__":
    # Load all three PDFs using the shared chunking module
    documents = load_all_pdfs()

    print(f"\nTotal PDF pages loaded: {len(documents)}")

    # Use the same chunking configuration for both models
    chunks = split_documents(
        documents,
        strategy="recursive",
        chunk_size=500,
        chunk_overlap=100,
    )

    print(f"Total chunks created: {len(chunks)}")

    # Build one vector store per embedding model
    for model_key, model_name in EMBEDDING_MODELS.items():
        create_vector_store(
            chunks=chunks,
            model_name=model_name,
            model_key=model_key,
        )

    print("\nEmbedding model setup completed.")
