
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# Each model has its own saved FAISS index.
EMBEDDING_MODELS = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "mpnet": "sentence-transformers/all-mpnet-base-v2",
}

DB_PATHS = {
    "minilm": "vectorstore/minilm",
    "mpnet": "vectorstore/mpnet",
}


def load_vector_store(model_key="minilm"):
    """
    Load the FAISS index associated with the selected embedding model.
    """
    if model_key not in EMBEDDING_MODELS:
        raise ValueError(
            f"Unknown model '{model_key}'. Choose from: "
            f"{', '.join(EMBEDDING_MODELS)}"
        )

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODELS[model_key],
        encode_kwargs={"normalize_embeddings": True},
    )

    vectorstore = FAISS.load_local(
        DB_PATHS[model_key],
        embeddings,
        allow_dangerous_deserialization=True,
    )

    return vectorstore


def retrieve_documents(query, vectorstore, k=3):
    """
    Return the top-k retrieved documents and their FAISS distances.
    Lower L2 distance generally indicates a closer match.
    """
    return vectorstore.similarity_search_with_score(query, k=k)


def display_results(results):
    """
    Print retrieved text with source, page, and distance.
    """
    if not results:
        print("No relevant documents were retrieved.")
        return

    print("\nTop retrieved results:\n")

    for rank, (doc, distance) in enumerate(results, start=1):
        source = doc.metadata.get("source", "Unknown source")
        page = doc.metadata.get("page", "Unknown page")

        # PyPDFLoader page metadata is normally zero-based.
        if isinstance(page, int):
            page = page + 1

        print(f"--- Result {rank} ---")
        print(f"Source: {source}")
        print(f"Page: {page}")
        print(f"L2 distance: {float(distance):.4f}")
        print(f"Text:\n{doc.page_content}")
        print()


if __name__ == "__main__":
    model_key = input(
        "Choose embedding model (minilm/mpnet) [minilm]: "
    ).strip().lower()

    if not model_key:
        model_key = "minilm"

    try:
        vectorstore = load_vector_store(model_key)
    except (ValueError, FileNotFoundError, OSError) as error:
        print(f"Could not load the vector store: {error}")
        raise SystemExit(1)

    print(f"\nLoaded model: {model_key}")
    print("Type 'exit' to stop.")

    while True:
        query = input("\nAsk a question: ").strip()

        if query.lower() == "exit":
            break

        if not query:
            print("Please enter a question.")
            continue

        results = retrieve_documents(query, vectorstore, k=3)
        display_results(results)
