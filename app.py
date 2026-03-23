import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DB_PATH = "vectorstore"

st.set_page_config(layout="wide", page_title="DocuMind AI", page_icon="📄")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700&family=Inter:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #eef0f3 !important; }
.block-container {
    padding-top: 1.5rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    max-width: 100% !important;
    padding-bottom: 0 !important;
}

/* Left dark panel */
div[data-testid="stVerticalBlock"]:has(div.left-marker) {
    background: #0d1b2a !important;
    border-radius: 18px !important;
    padding: 32px 28px !important;
    min-height: 92vh !important;
}

/* Right white card */
div[data-testid="stVerticalBlock"]:has(div.right-marker) {
    background: #ffffff !important;
    border-radius: 18px !important;
    padding: 0px !important;
    min-height: 92vh !important;
    box-shadow: 0 4px 32px rgba(0,0,0,0.07) !important;
    display: flex !important;
    flex-direction: column !important;
}

/* Suggestion buttons */
div[data-testid="stVerticalBlock"]:has(div.right-marker) .stButton > button {
    background: #f8f6ff !important;
    border: 1.5px solid #ddd5ff !important;
    border-radius: 22px !important;
    color: #4a3fbf !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 13px !important;
    padding: 8px 18px !important;
    font-weight: 500 !important;
    width: 100% !important;
    box-shadow: none !important;
}
div[data-testid="stVerticalBlock"]:has(div.right-marker) .stButton > button:hover {
    background: #ede8ff !important;
    border-color: #b8aaff !important;
    color: #2e24a0 !important;
}

/* Chat input styling */
[data-testid="stChatInput"] {
    background: #f5f3ff !important;
    border-radius: 14px !important;
    border: 1.5px solid #ddd5ff !important;
}
[data-testid="stChatInput"] textarea {
    color: #f5f3ff !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 14px !important;
    background: transparent !important;
}

/* The scrollable chat history box */
.chat-scroll-box {
    height: 380px;
    overflow-y: auto;
    padding: 16px 24px;
    display: flex;
    flex-direction: column;
    gap: 10px;
    border-top: 1px solid #f0ecff;
    border-bottom: 1px solid #f0ecff;
    margin: 0 0 12px 0;
    scroll-behavior: smooth;
}
.chat-scroll-box::-webkit-scrollbar { width: 5px; }
.chat-scroll-box::-webkit-scrollbar-track { background: #f8f6ff; border-radius: 10px; }
.chat-scroll-box::-webkit-scrollbar-thumb { background: #d4ccff; border-radius: 10px; }
.chat-scroll-box::-webkit-scrollbar-thumb:hover { background: #b0a4f5; }

/* Chat bubble styles */
.bubble-row-user {
    display: flex;
    justify-content: flex-end;
    margin-bottom: 4px;
}
.bubble-row-ai {
    display: flex;
    justify-content: flex-start;
    margin-bottom: 4px;
}
.bubble-user {
    background: #0d1b2a;
    color: #d8eaf8;
    border-radius: 16px 16px 4px 16px;
    padding: 11px 16px;
    font-size: 13.5px;
    line-height: 1.6;
    max-width: 78%;
    font-family: 'Inter', sans-serif;
}
.bubble-ai {
    background: #f3f0ff;
    color: #1a1635;
    border: 1px solid #e4dcff;
    border-radius: 16px 16px 16px 4px;
    padding: 11px 16px;
    font-size: 13.5px;
    line-height: 1.6;
    max-width: 82%;
    font-family: 'Inter', sans-serif;
}
.bubble-avatar {
    width: 28px;
    height: 28px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 11px;
    font-weight: 600;
    flex-shrink: 0;
    margin: 0 7px;
    align-self: flex-end;
}
.avatar-user { background: #0d1b2a; color: #d8eaf8; }
.avatar-ai   { background: linear-gradient(135deg,#634dff,#14b8a6); color: #fff; }

.typing-indicator {
    display: flex; gap: 5px; align-items: center;
    padding: 11px 16px;
    background: #f3f0ff;
    border: 1px solid #e4dcff;
    border-radius: 16px 16px 16px 4px;
    width: fit-content;
}
.td {
    width: 7px; height: 7px; border-radius: 50%;
    background: #9b7fe8;
    animation: td 1.2s infinite;
}
.td:nth-child(2){animation-delay:.2s;}
.td:nth-child(3){animation-delay:.4s;}
@keyframes td{0%,80%,100%{transform:translateY(0);}40%{transform:translateY(-6px);}}

.empty-chat {
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    height: 100%; gap: 8px; opacity: 0.4;
    font-size: 13px; color: #888;
}

#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Load vector store ─────────────────────────────────────────────
@st.cache_resource
def load_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)

vectorstore = load_vector_store()

# ── Session state ─────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "query" not in st.session_state:
    st.session_state.query = ""
if "show_suggestions" not in st.session_state:
    st.session_state.show_suggestions = True  # only show suggestions when chat is empty


# ── Layout ────────────────────────────────────────────────────────
col_left, col_right = st.columns([4, 6], gap="medium")


# ════════════════════════════════════════
# LEFT PANEL
# ════════════════════════════════════════
with col_left:
    st.markdown('<div class="left-marker"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="margin-bottom:8px;">
        <span style="font-family:'Sora',sans-serif;font-size:21px;font-weight:700;color:#fff;">
            📄 Docu<span style="color:#14b8a6;">Mind</span> AI
        </span>
    </div>
    <div style="font-family:'Sora',sans-serif;font-size:27px;font-weight:700;
                color:#fff;line-height:1.3;margin:16px 0 12px;">
        Your documents,<br>
        <span style="color:#a78bfa;">instantly answered.</span>
    </div>
    <div style="font-size:13.5px;color:#7a9ab5;line-height:1.75;margin-bottom:28px;">
        An AI system that answers questions about company documents —
        policies, manuals, and technical guides — in seconds.
    </div>
    <div style="display:flex;flex-direction:column;gap:11px;margin-bottom:36px;">
        <div style="display:flex;align-items:center;gap:10px;">
            <div style="width:9px;height:9px;border-radius:50%;background:#a78bfa;"></div>
            <span style="font-size:13px;color:#c8d6e3;font-weight:500;">Semantic document search</span>
        </div>
        <div style="display:flex;align-items:center;gap:10px;">
            <div style="width:9px;height:9px;border-radius:50%;background:#14b8a6;"></div>
            <span style="font-size:13px;color:#c8d6e3;font-weight:500;">Policy &amp; manual intelligence</span>
        </div>
        <div style="display:flex;align-items:center;gap:10px;">
            <div style="width:9px;height:9px;border-radius:50%;background:#ec4899;"></div>
            <span style="font-size:13px;color:#c8d6e3;font-weight:500;">Instant, cited answers</span>
        </div>
    </div>
    <div style="border-top:1px solid rgba(255,255,255,0.07);padding-top:18px;">
        <div style="font-size:10.5px;color:#3d5a70;text-transform:uppercase;
                    letter-spacing:0.07em;margin-bottom:10px;">Powered by</div>
        <div style="display:flex;flex-wrap:wrap;gap:7px;">
            <span style="background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.1);
                         color:#7a9ab5;font-size:11.5px;padding:4px 12px;border-radius:20px;">LangChain</span>
            <span style="background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.1);
                         color:#7a9ab5;font-size:11.5px;padding:4px 12px;border-radius:20px;">FAISS</span>
            <span style="background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.1);
                         color:#7a9ab5;font-size:11.5px;padding:4px 12px;border-radius:20px;">Streamlit</span>
            <span style="background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.1);
                         color:#7a9ab5;font-size:11.5px;padding:4px 12px;border-radius:20px;">HuggingFace</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════
# RIGHT PANEL
# ════════════════════════════════════════
with col_right:
    st.markdown('<div class="right-marker"></div>', unsafe_allow_html=True)

    # ── Header ──
    st.markdown("""
    <div style="display:flex;align-items:center;justify-content:space-between;
                padding:22px 24px 14px;border-bottom:1px solid #f0ecff;">
        <div style="display:flex;align-items:center;gap:10px;">
            <div style="width:9px;height:9px;border-radius:50%;background:#22c55e;
                        box-shadow:0 0 0 3px rgba(34,197,94,0.15);"></div>
            <div>
                <div style="font-family:'Sora',sans-serif;font-size:15px;
                            font-weight:600;color:#111;">Ask DocuMind AI</div>
                <div style="font-size:12px;color:#999;margin-top:2px;">
                    Company knowledge, at your fingertips</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Build the scrollable chat HTML ──
    chat_html = '<div class="chat-scroll-box" id="chatbox">'

    if not st.session_state.messages:
        chat_html += """
        <div class="empty-chat">
            <svg width="32" height="32" viewBox="0 0 24 24" fill="none"
                 stroke="#ccc" stroke-width="1.5">
                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                <polyline points="14 2 14 8 20 8"/>
                <line x1="9" y1="13" x2="15" y2="13"/>
            </svg>
            <span>Ask a question to get started</span>
        </div>
        """
    else:
        for msg in st.session_state.messages:
            # Escape HTML in message content
            content = (msg["content"]
                       .replace("&", "&amp;")
                       .replace("<", "&lt;")
                       .replace(">", "&gt;")
                       .replace("\n", "<br>"))
            if msg["role"] == "user":
                chat_html += f"""
                <div class="bubble-row-user">
                    <div class="bubble-user">{content}</div>
                    <div class="bubble-avatar avatar-user">U</div>
                </div>"""
            else:
                chat_html += f"""
                <div class="bubble-row-ai">
                    <div class="bubble-avatar avatar-ai">AI</div>
                    <div class="bubble-ai">{content}</div>
                </div>"""

    chat_html += '</div>'

    # Auto-scroll to bottom JS
    chat_html += """
    <script>
        const box = document.getElementById('chatbox');
        if (box) box.scrollTop = box.scrollHeight;
    </script>
    """

    st.markdown(chat_html, unsafe_allow_html=True)

    # ── Suggestions — only show when no conversation yet ──
    if st.session_state.show_suggestions:
        st.markdown("""
        <div style="font-size:11px;color:#bbb;text-transform:uppercase;
                    letter-spacing:0.06em;margin:0 24px 10px;">Suggested questions</div>
        """, unsafe_allow_html=True)

        b1, b2 = st.columns(2)
        with b1:
            if st.button("What is the leave policy?", key="q1", use_container_width=True):
                st.session_state.query = "What is the company leave policy?"
                st.session_state.show_suggestions = False
            st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
            if st.button("What are the security policies?", key="q2", use_container_width=True):
                st.session_state.query = "What are the company security policies?"
                st.session_state.show_suggestions = False
        with b2:
            if st.button("What is the code of conduct?", key="q3", use_container_width=True):
                st.session_state.query = "What is the code of conduct?"
                st.session_state.show_suggestions = False
            st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
            if st.button("Where is the employee handbook?", key="q4", use_container_width=True):
                st.session_state.query = "Where can I find the employee handbook?"
                st.session_state.show_suggestions = False

    # Padding before input
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # ── Chat input — always at the bottom ──
    query = st.chat_input("Ask about policies, manuals, guides...")

    if not query and st.session_state.query:
        query = st.session_state.query
        st.session_state.query = ""

    # ── Process query ──
    if query:
        st.session_state.show_suggestions = False
        st.session_state.messages.append({"role": "user", "content": query})

        with st.spinner("Searching documents..."):
            results = vectorstore.similarity_search(query, k=3)
            answer = "\n\n".join([doc.page_content for doc in results])
            if not answer.strip():
                answer = "I couldn't find relevant information. Please ensure your documents are loaded in the vector store."

        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.rerun()