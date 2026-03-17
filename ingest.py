from langchain_community.document_loaders import PyPDFLoader
import os

def load_all_pdfs(data_path):
    documents = []

    for file in os.listdir(data_path):
        if file.endswith(".pdf"):
            file_path = os.path.join(data_path, file)
            print(f"Loading: {file}")

            loader = PyPDFLoader(file_path)
            docs = loader.load()
            documents.extend(docs)

    return documents


if __name__ == "__main__":
    DATA_PATH = r"/workspaces/AI-Knowledge-Base-Assistant/data"
    docs = load_all_pdfs(DATA_PATH)

    print("\n✅ Total pages loaded:", len(docs))

    print("\n📄 Sample content:\n")
    for i, doc in enumerate(docs[:3]):  # show first 3 pages only
        print(f"--- Document {i+1} ---")
        print(doc.page_content[:500])  # first 500 characters
        print("\n")