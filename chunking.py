from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
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


def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = text_splitter.split_documents(documents)
    return chunks


if __name__ == "__main__":
    DATA_PATH = "/workspaces/AI-Knowledge-Base-Assistant/data"

    docs = load_all_pdfs(DATA_PATH)
    print(f"\n📄 Total pages loaded: {len(docs)}")

    chunks = split_documents(docs)
    print(f"\n✂️ Total chunks created: {len(chunks)}")

    print("\n🔍 Sample chunk:\n")
    print(chunks[0].page_content)
    print(f"\n📊 Total chunks: {len(chunks)}")
print(f"📊 Average chunks per document: {len(chunks)/len(docs):.2f}")

    