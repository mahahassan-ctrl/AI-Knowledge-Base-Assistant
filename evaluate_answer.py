import json
import os
import time
from pathlib import Path

from google import genai
from chatbot import generate_answer, GEMINI_MODEL
from retrieval import load_vector_store, retrieve_documents

# Conservative limits to avoid exhausting the API quota.
MAX_NEW_ATTEMPTS = 5
REQUEST_DELAY_SECONDS = 15

REFUSAL_QUESTIONS = [
    "What is the company's annual revenue?",
    "Who is the company's current chief executive officer?",
    "What is the company's current stock market price?",
    "How many customers does the company have worldwide?",
    "What is the company's market share in its industry?",
    "Which country has the highest employee satisfaction score?",
    "What was the company's profit last financial year?",
    "What is the company's five-year share price forecast?",
    "Which competitor has the largest market share?",
    "What is the company's current advertising budget?",
    "What is the exact salary of the company's chief executive officer?",
    "Which employee received the highest performance rating last year?",
    "What is the company's current customer churn rate?",
    "What is the company's latest quarterly revenue growth?",
    "Which stock exchange is the company listed on?",
    "What is the company's current cybersecurity insurance premium?",
    "How many products did the company sell last month?",
    "What is the company's current net profit margin?",
    "Which country will the company expand into next year?",
    "What is the company's current valuation?",
]


def save_results(path, output):
    """Save all results so progress survives interruptions."""
    with path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)


def is_quota_error(error):
    message = str(error).lower()
    return (
        "429" in message
        or "resource_exhausted" in message
        or "quota exceeded" in message
        or "rate limit" in message
    )


def main():
    if not os.getenv("GEMINI_API_KEY"):
        raise SystemExit("GEMINI_API_KEY is not set.")

    questions_path = Path("evaluation/retrieval_eval_questions.json")
    results_path = Path("evaluation/results/answer_quality_outputs.json")
    results_path.parent.mkdir(parents=True, exist_ok=True)

    with questions_path.open(encoding="utf-8") as f:
        data = json.load(f)

    tests = [
        {
            "id": q["id"],
            "type": "answerable",
            "question": q["question"],
            "expected_pages": q["relevant_pages"],
        }
        for q in data["questions"]
    ]

    tests.extend(
        {
            "id": f"R{i:02d}",
            "type": "unsupported",
            "question": question,
            "expected_pages": [],
        }
        for i, question in enumerate(REFUSAL_QUESTIONS, start=1)
    )

    # Load previous results instead of starting from scratch.
    if results_path.exists():
        with results_path.open(encoding="utf-8") as f:
            previous = json.load(f)
        existing = {
            result["id"]: result
            for result in previous.get("results", [])
        }
        print(f"Loaded {len(existing)} previously recorded results.")
    else:
        existing = {}
        print("No previous results found; starting a new evaluation.")

    # Preserve prior answers, errors, and manual review fields.
    results = {}
    for test in tests:
        old = existing.get(test["id"], {})
        results[test["id"]] = {
            **test,
            **old,
            **test,
            "answer": old.get("answer", ""),
            "error": old.get("error"),
            "manual_answer_correct": old.get("manual_answer_correct"),
            "manual_refusal_correct": old.get("manual_refusal_correct"),
        }

    def completed(result):
        return bool(result.get("answer", "").strip()) and not result.get("error")

    pending = [
        test for test in tests
        if not completed(results[test["id"]])
    ]

    already_done = len(tests) - len(pending)
    print(f"Already successful: {already_done}")
    print(f"Questions needing retry: {len(pending)}")
    print(f"Maximum new attempts this run: {MAX_NEW_ATTEMPTS}")

    if not pending:
        print("All questions already have successful responses.")
        return

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    vectorstore = load_vector_store("minilm")
    attempts = 0

    for test in pending:
        if attempts >= MAX_NEW_ATTEMPTS:
            print("\nAttempt limit reached. Run the script again later to continue.")
            break

        attempts += 1
        result = results[test["id"]]

        if attempts > 1:
            time.sleep(REQUEST_DELAY_SECONDS)

        print(f"\nAttempt {attempts}/{MAX_NEW_ATTEMPTS}: "
              f"{test['id']} — {test['question']}")

        try:
            documents = retrieve_documents(
                query=test["question"],
                vectorstore=vectorstore,
                k=3,
            )
            answer = generate_answer(client, test["question"], documents)

            result["answer"] = answer
            result["error"] = None
            print("Response saved.")

        except Exception as e:
            result["answer"] = ""
            result["error"] = f"{type(e).__name__}: {e}"
            print(f"Request failed: {type(e).__name__}: {e}")

            # Stop immediately on quota/rate-limit errors.
            save_results(
                results_path,
                {
                    "model": GEMINI_MODEL,
                    "embedding_model": "minilm",
                    "answerable_count": len(data["questions"]),
                    "unsupported_count": len(REFUSAL_QUESTIONS),
                    "results": [results[t["id"]] for t in tests],
                },
            )

            if is_quota_error(e):
                print(
                    "\nQuota or rate limit reached. Stopping safely; "
                    "previous successful answers have been preserved."
                )
                break

        output = {
            "model": GEMINI_MODEL,
            "embedding_model": "minilm",
            "answerable_count": len(data["questions"]),
            "unsupported_count": len(REFUSAL_QUESTIONS),
            "results": [results[t["id"]] for t in tests],
        }
        save_results(results_path, output)

    # Final save, including results that were not retried.
    output = {
        "model": GEMINI_MODEL,
        "embedding_model": "minilm",
        "answerable_count": len(data["questions"]),
        "unsupported_count": len(REFUSAL_QUESTIONS),
        "results": [results[t["id"]] for t in tests],
    }
    save_results(results_path, output)

    successful = sum(completed(r) for r in output["results"])
    failed = len(output["results"]) - successful

    print("\nEvaluation progress saved.")
    print(f"Successful responses: {successful}/{len(tests)}")
    print(f"Questions still needing attention: {failed}")
    print(f"File: {results_path}")


if __name__ == "__main__":
    main()