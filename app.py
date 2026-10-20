

import os
import streamlit as st
from google import genai

from retrieval import load_vector_store, retrieve_documents
from chatbot import generate_answer

# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="DocuMind AI",
    page_icon="📄",
    layout="wide",
)

TOP_K = 3

EMBEDDING_MODELS = {
    "MiniLM": "minilm",
    "MPNet": "mpnet",
}

# --------------------------------------------------
# Styling
# --------------------------------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700&family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: #eef0f3;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 1rem;
        max-width: 1400px;
    }

    /* Main panels */
    div[data-testid="stVerticalBlock"]:has(.left-marker) {
        background: #0d1b2a;
        border-radius: 18px;
        padding: 28px;
        min-height: 85vh;
    }

    div[data-testid="stVerticalBlock"]:has(.right-marker) {
        background: #ffffff;
        border-radius: 18px;
        padding: 24px;
        min-height: 85vh;
        box-shadow: 0 4px 32px rgba(0, 0, 0, 0.07);
    }

    /* Make chat questions and answers black */
    [data-testid="stChatMessage"],
    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] li,
    [data-testid="stChatMessage"] h1,
    [data-testid="stChatMessage"] h2,
    [data-testid="stChatMessage"] h3,
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
        color: #000000 !important;
    }

    /* Make typed question text black */
    [data-testid="stChatInput"] textarea {
        color: #000000 !important;
        -webkit-text-fill-color: #000000 !important;
        background: transparent !important;
        font-family: 'Inter', sans-serif;
        font-size: 14px;
    }

    [data-testid="stChatInput"] textarea::placeholder {
        color: #555555 !important;
        -webkit-text-fill-color: #555555 !important;
    }

    [data-testid="stChatInput"] {
        background: #f5f3ff !important;
        border: 1.5px solid #ddd5ff !important;
        border-radius: 14px !important;
    }

    /* Suggested question buttons */
    .stButton > button {
        background: #f8f6ff;
        border: 1px solid #ddd5ff;
        border-radius: 12px;
        color: #000000 !important;
        width: 100%;
        min-height: 48px;
    }

    .stButton > button:hover {
        background: #ede8ff;
        border-color: #b8aaff;
        color: #000000 !important;
    }

    #MainMenu, footer {
        visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# Session state
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

if "show_suggestions" not in st.session_state:
    st.session_state.show_suggestions = True

if "embedding_choice" not in st.session_state:
    st.session_state.embedding_choice = "MiniLM"

# --------------------------------------------------
# Cached resources
# --------------------------------------------------

@st.cache_resource
def get_vector_store(model_key):
    return load_vector_store(model_key)


@st.cache_resource
def get_gemini_client(api_key):
    return genai.Client(api_key=api_key)


# --------------------------------------------------
# Layout
# --------------------------------------------------

col_left, col_right = st.columns(
    [4, 6],
    gap="medium",
)

# --------------------------------------------------
# Left panel
# --------------------------------------------------

with col_left:
    st.markdown(
        '<div class="left-marker"></div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div style="margin-bottom: 24px;">
            <span style="font-family:'Sora',sans-serif;
                         font-size:24px;font-weight:700;color:#ffffff;">
                📄 Docu<span style="color:#14b8a6;">Mind</span> AI
            </span>
        </div>

        <div style="font-family:'Sora',sans-serif;
                    font-size:28px;font-weight:700;
                    color:#ffffff;line-height:1.35;
                    margin-bottom:16px;">
            Your documents,<br>
            <span style="color:#a78bfa;">
                instantly answered.
            </span>
        </div>

        <div style="font-size:14px;color:#c8d6e3;
                    line-height:1.8;margin-bottom:30px;">
            Ask questions about company documents, policies,
            manuals, and security guidelines.
        </div>

        <div style="font-size:14px;color:#c8d6e3;
                    line-height:2.4;">
            <div>🟣 Semantic document search</div>
            <div>🟢 Policy and manual intelligence</div>
            <div>🔵 AI-generated answers with citations</div>
        </div>

        <div style="margin-top:35px;padding-top:20px;
                    border-top:1px solid rgba(255,255,255,0.15);
                    font-size:12px;color:#9eb3c7;line-height:2;">
            Powered by<br>
            LangChain · FAISS · Hugging Face · Gemini
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    st.markdown(
        "**Embedding model**",
        help="Choose which saved vector index to search.",
    )

    st.selectbox(
        "Choose an embedding model",
        options=list(EMBEDDING_MODELS.keys()),
        key="embedding_choice",
        label_visibility="collapsed",
    )

    st.caption(
        "Both indexes must already exist in the vectorstore folder."
    )

# --------------------------------------------------
# Right panel
# --------------------------------------------------

with col_right:
    st.markdown(
        '<div class="right-marker"></div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div style="padding:8px 0 18px;
                    border-bottom:1px solid #f0ecff;
                    margin-bottom:18px;">
            <div style="font-family:'Sora',sans-serif;
                        font-size:20px;font-weight:600;color:#111111;">
                Ask DocuMind AI
            </div>
            <div style="font-size:13px;color:#666666;margin-top:5px;">
                Company knowledge, at your fingertips
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Display conversation history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Suggested questions before the first message
    if (
        st.session_state.show_suggestions
        and not st.session_state.messages
    ):
        st.markdown("**Suggested questions**")

        suggestions = [
            "What is the employee leave policy?",
            "What are the security policies?",
            "What is the code of conduct?",
            "How should an employee report a security incident?",
        ]

        suggestion_columns = st.columns(2)

        for index, suggestion in enumerate(suggestions):
            with suggestion_columns[index % 2]:
                if st.button(
                    suggestion,
                    key=f"suggestion_{index}",
                    use_container_width=True,
                ):
                    st.session_state.pending_query = suggestion
                    st.session_state.show_suggestions = False
                    st.rerun()

    # Question input
    typed_query = st.chat_input(
        "Ask about policies, manuals, and guides..."
    )

    query = typed_query

    if query is None and st.session_state.pending_query:
        query = st.session_state.pending_query
        st.session_state.pending_query = None

    # --------------------------------------------------
    # Process question
    # --------------------------------------------------

    if query:
        query = query.strip()

        if query:
            st.session_state.show_suggestions = False

            st.session_state.messages.append(
                {
                    "role": "user",
                    "content": query,
                }
            )

            with st.chat_message("assistant"):
                with st.spinner(
                    "Searching documents and generating an answer..."
                ):
                    api_key = os.getenv("GEMINI_API_KEY")

                    if not api_key:
                        answer = (
                            "Gemini API key not found. Set the "
                            "GEMINI_API_KEY environment variable "
                            "in your Codespace, then restart the app."
                        )
                    else:
                        try:
                            model_key = EMBEDDING_MODELS[
                                st.session_state.embedding_choice
                            ]

                            vectorstore = get_vector_store(model_key)
                            client = get_gemini_client(api_key)

                            results = retrieve_documents(
                                query=query,
                                vectorstore=vectorstore,
                                k=TOP_K,
                            )

                            answer = generate_answer(
                                client=client,
                                query=query,
                                results=results,
                            )

                        except Exception as error:
                            answer = (
                                "Sorry, something went wrong while "
                                "processing your question. "
                                f"Error type: {type(error).__name__}. "
                                "Check the Codespaces terminal for details."
                            )

                st.markdown(answer)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                }
            )

            st.rerun()
