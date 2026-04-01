import streamlit as st
import sys
from pathlib import Path

# add src folder to path
sys.path.append(str(Path(__file__).parent / "src"))

from rag_pipeline import RAGPipeline

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Quantum RAG Chatbot", page_icon="⚛️")

st.title("⚛️ Quantum Computing RAG Chatbot")
st.write("Ask questions about quantum computing!")

# -------------------------------
# LOAD RAG PIPELINE (CACHE)
# -------------------------------
@st.cache_resource
def load_rag():
    return RAGPipeline(dataset_dir="dataset")

rag = load_rag()

# -------------------------------
# SESSION STATE (CHAT HISTORY)
# -------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# -------------------------------
# DISPLAY CHAT HISTORY
# -------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -------------------------------
# USER INPUT
# -------------------------------
user_input = st.chat_input("Ask a question...")

if user_input:
    # Save user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = rag.query(user_input)
            answer = result["answer"]

            st.markdown(answer)

            # Show sources
            st.markdown("### 📚 Sources")
            for r in result["retrieved_chunks"][:3]:
                st.markdown(f"- **{r.chunk.doc_title}** (score: {r.score:.3f})")

    # Save assistant response
    st.session_state.messages.append({"role": "assistant", "content": answer})