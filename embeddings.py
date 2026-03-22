from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

DATA_PATH = "data"
DB_PATH = "vectorstore"

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
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    return splitter.split_documents(documents)


def create_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(DB_PATH)

    return vectorstore


if __name__ == "__main__":
    docs = load_all_pdfs(DATA_PATH)
    print(f"\n📄 Total pages: {len(docs)}")

    chunks = split_documents(docs)
    print(f"✂️ Total chunks: {len(chunks)}")

    vectorstore = create_vector_store(chunks)

    print("\n✅ Embeddings created and saved (LOCAL MODEL)")