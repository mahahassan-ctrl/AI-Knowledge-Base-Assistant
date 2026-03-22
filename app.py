import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DB_PATH = "vectorstore"

@st.cache_resource
def load_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vectorstore = FAISS.load_local(
        DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vectorstore


def generate_answer(query, docs):
    answer = "📄 Based on the documents:\n\n"

    for doc in docs:
        answer += doc.page_content + "\n\n"

    return answer


# UI
st.set_page_config(page_title="AI Knowledge Assistant", layout="wide")

st.title("🤖 AI Knowledge Base Assistant")
st.write("Ask questions about company documents")

query = st.text_input("💬 Enter your question:")

if query:
    vectorstore = load_vector_store()
    results = vectorstore.similarity_search(query, k=3)

    answer = generate_answer(query, results)

    st.subheader("📌 Answer")
    st.write(answer)

    st.subheader("📚 Source Chunks")
    for i, doc in enumerate(results):
        st.write(f"**Result {i+1}:**")
        st.write(doc.page_content)
        