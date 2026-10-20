
import csv
import json
from pathlib import Path

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from chunking import load_all_pdfs, split_documents


EVALUATION_FILE = Path("evaluation/retrieval_eval_questions.json")
RESULTS_DIR = Path("evaluation/results")

EMBEDDING_MODELS = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "mpnet": "sentence-transformers/all-mpnet-base-v2",
}

CHUNKING_STRATEGIES = ["recursive", "fixed", "structure"]
TOP_K = 3


def normalize_source(source):
    """Normalize PDF metadata to a filename."""
    if not source:
        return ""
    return Path(str(source)).name


def normalize_page(metadata):
    """Convert LangChain's zero-based PDF page to a 1-based page."""
    page = metadata.get("page")

    if page is None:
        return None

    try:
        return int(page) + 1
    except (TypeError, ValueError):
        return None


def load_evaluation_questions():
    if not EVALUATION_FILE.exists():
        raise FileNotFoundError(
            f"Evaluation dataset not found: {EVALUATION_FILE}\n"
            "Place retrieval_eval_questions.json inside evaluation/."
        )

    with EVALUATION_FILE.open("r", encoding="utf-8") as file:
        dataset = json.load(file)

    questions = dataset.get("questions", [])

    if not questions:
        raise ValueError("The evaluation dataset contains no questions.")

    if len(questions) != 40:
        raise ValueError(
            f"Expected 40 questions, but found {len(questions)}."
        )

    return questions


def page_is_relevant(document, relevant_pages):
    """Return True if a retrieved chunk matches a labelled PDF page."""
    source = normalize_source(document.metadata.get("source"))
    page = normalize_page(document.metadata)

    for relevant in relevant_pages:
        expected_source = normalize_source(relevant.get("source"))
        expected_page = relevant.get("page")

        if (
            source == expected_source
            and page == int(expected_page)
        ):
            return True

    return False


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    questions = load_evaluation_questions()

    print(f"Loaded {len(questions)} evaluation questions.")
    print("Loading the three PDF documents...")

    documents = load_all_pdfs()

    print(f"Loaded {len(documents)} PDF pages.")
    print("\nStarting the six benchmark runs.")
    print("The first run may take longer while models load.\n")

    summary = []
    detailed_results = []

    for strategy in CHUNKING_STRATEGIES:
        print(f"\n{'=' * 60}")
        print(f"CHUNKING STRATEGY: {strategy}")
        print(f"{'=' * 60}")

        chunks = split_documents(
            documents,
            strategy=strategy,
            chunk_size=500,
            chunk_overlap=100,
        )

        print(f"Created {len(chunks)} chunks.")

        for model_key, model_name in EMBEDDING_MODELS.items():
            print(f"\nEmbedding model: {model_key}")

            embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                encode_kwargs={"normalize_embeddings": True},
            )

            vectorstore = FAISS.from_documents(chunks, embeddings)

            hits = 0

            for item in questions:
                question_id = item["id"]
                query = item["question"]
                relevant_pages = item["relevant_pages"]

                results = vectorstore.similarity_search_with_score(
                    query,
                    k=TOP_K,
                )

                retrieved = []

                for rank, (document, distance) in enumerate(results, start=1):
                    source = normalize_source(
                        document.metadata.get("source")
                    )
                    page = normalize_page(document.metadata)
                    relevant = page_is_relevant(
                        document,
                        relevant_pages,
                    )

                    retrieved.append({
                        "rank": rank,
                        "source": source,
                        "page": page,
                        "distance": float(distance),
                        "relevant_page_match": relevant,
                    })

                hit = any(
                    result["relevant_page_match"]
                    for result in retrieved
                )

                hits += int(hit)

                detailed_results.append({
                    "strategy": strategy,
                    "embedding_model": model_key,
                    "question_id": question_id,
                    "question": query,
                    "hit_at_3": hit,
                    "relevant_pages": relevant_pages,
                    "retrieved": retrieved,
                })

            recall = hits / len(questions) * 100

            summary.append({
                "strategy": strategy,
                "embedding_model": model_key,
                "chunk_count": len(chunks),
                "questions": len(questions),
                "hits_at_3": hits,
                "misses_at_3": len(questions) - hits,
                "recall_at_3_percent": round(recall, 2),
            })

            print(
                f"Recall@3: {hits}/{len(questions)} "
                f"= {recall:.2f}%"
            )

            del vectorstore
            del embeddings

    summary_path = RESULTS_DIR / "benchmark_summary.csv"
    details_path = RESULTS_DIR / "benchmark_details.json"

    with summary_path.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "strategy",
                "embedding_model",
                "chunk_count",
                "questions",
                "hits_at_3",
                "misses_at_3",
                "recall_at_3_percent",
            ],
        )
        writer.writeheader()
        writer.writerows(summary)

    with details_path.open("w", encoding="utf-8") as file:
        json.dump(detailed_results, file, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print("FINAL RESULTS")
    print(f"{'=' * 60}")

    for result in sorted(
        summary,
        key=lambda row: row["recall_at_3_percent"],
        reverse=True,
    ):
        print(
            f"{result['strategy']:10} | "
            f"{result['embedding_model']:6} | "
            f"{result['hits_at_3']:2}/{result['questions']} | "
            f"{result['recall_at_3_percent']:6.2f}%"
        )

    print("\nSaved benchmark files:")
    print(f"  {summary_path}")
    print(f"  {details_path}")


if __name__ == "__main__":
    main()
