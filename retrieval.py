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


if __name__ == "__main__":
    vectorstore = load_vector_store()

    while True:
        query = input("\n💬 Ask a question (or type 'exit'): ")

        if query.lower() == "exit":
            break

        results = vectorstore.similarity_search(query, k=3)

        print("\n🔍 Top Results:\n")

        for i, doc in enumerate(results):
            print(f"--- Result {i+1} ---")
            print(doc.page_content)
            print("\n")