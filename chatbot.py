from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DB_PATH = "vectorstore"

def load_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vectorstore = FAISS.load_local(
        DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    return vectorstore


def generate_answer(query, docs):
    answer = "Based on the documents:\n\n"

    for doc in docs:
        answer += doc.page_content + "\n\n"

    return answer


if __name__ == "__main__":
    vectorstore = load_vector_store()

    while True:
        query = input("\n💬 Ask your question (or type 'exit'): ")

        if query.lower() == "exit":
            break

        results = vectorstore.similarity_search(query, k=3)

        answer = generate_answer(query, results)

        print("\n🤖 Answer:\n")
        print(answer)
        