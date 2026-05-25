"""
app/streamlit_app.py
---------------------
Streamlit frontend for SR-RAG.

Run from project root:
    streamlit run app/streamlit_app.py

CRITICAL: @st.cache_resource is used on all heavy objects
(pipeline, embedding model, ChromaDB). Without it, Streamlit
reloads them on every interaction, making every query 10-20s.
"""

import sys
from pathlib import Path

# Ensure project root is on path when launched from app/ directory
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import json
from src.pipeline import SRRagPipeline

# ---------------------------------------------------------------------------
# Page config — must be first Streamlit call
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="SR-RAG | Super-Resolution Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Cached resource loaders — loaded ONCE, reused across all reruns
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading pipeline (first run takes ~30s)...")
def get_pipeline():
    """
    Loads embedding model + ChromaDB collection once and pins in memory.
    Subsequent reruns skip this entirely.
    """
    return SRRagPipeline(
        persist_dir="vector_store",
        collection_name="sr_papers",
    )


@st.cache_data(show_spinner=False)
def load_corpus_metadata():
    path = Path("data/processed/paper_metadata.csv")
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# Example questions
# ---------------------------------------------------------------------------

EXAMPLES = [
    "What loss function does SRGAN use?",
    "How does SwinIR use transformer attention?",
    "Compare SRGAN and ESRGAN architectures.",
    "Which papers report results on DIV2K?",
    "What is residual channel attention in RCAN?",
    "How does EDSR differ from SRResNet?",
    "What makes RealESRGAN handle real-world images?",
    "Which methods use adversarial training?",
]

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar(corpus_df):
    with st.sidebar:
        st.title("⚙️ Settings")

        top_k = st.slider(
            "Chunks to retrieve (top_k)",
            min_value=3, max_value=15, value=5,
            help="More chunks = more context but slower and costlier."
        )

        show_raw = st.checkbox(
            "Show raw retrieved chunks",
            value=False,
            help="Debug view — shows every chunk sent to the LLM."
        )

        show_tokens = st.checkbox(
            "Show token usage",
            value=True,
        )

        st.divider()
        st.subheader("📚 Corpus")

        if not corpus_df.empty:
            for _, row in corpus_df.iterrows():
                method = row.get("method", row.get("file_name", "?"))
                year   = row.get("year", "")
                url    = row.get("source_url", "")
                if url:
                    st.markdown(f"**[{method}]({url})** ({year})")
                else:
                    st.markdown(f"**{method}** ({year})")
        else:
            st.info("Metadata not found.")

        st.divider()
        st.caption("SR-RAG · Week 1 build")

    return top_k, show_raw, show_tokens


# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------

def main():
    pipeline   = get_pipeline()
    corpus_df  = load_corpus_metadata()
    top_k, show_raw, show_tokens = render_sidebar(corpus_df)

    # Header
    st.title("🔬 SR-RAG")
    st.caption("Ask questions across 10 image super-resolution research papers.")

    col_info, _ = st.columns([3, 1])
    with col_info:
        info = pipeline.info()
        st.caption(
            f"Collection: **{info['total_chunks']}** chunks · "
            f"Model: **{info['embedding_model'].split('/')[-1]}** · "
            f"LLM: **{info['llm_provider']}**"
        )

    st.divider()

    # Example question buttons
    st.subheader("Try an example")
    cols = st.columns(4)
    selected = None
    for i, example in enumerate(EXAMPLES):
        if cols[i % 4].button(example, use_container_width=True, key=f"ex_{i}"):
            selected = example

    st.divider()

    # Query input
    query = st.text_input(
        "Or ask your own question:",
        value=selected or st.session_state.get("last_query", ""),
        placeholder="e.g. What PSNR does EDSR report on Set5?",
        key="query_input",
    )

    ask_clicked = st.button("Ask", type="primary", use_container_width=False)

    # Run query
    if (ask_clicked or selected) and query.strip():
        st.session_state["last_query"] = query

        with st.spinner("Retrieving and generating answer..."):
            result = pipeline.query(
                question=query,
                top_k=top_k,
                include_raw_chunks=show_raw,
            )

        # Answer
        st.subheader("Answer")
        st.markdown(result["answer"])

        # Latency + token usage
        meta_parts = [f"⏱ {result['total_ms']:.0f} ms total"]
        if show_tokens:
            t = result["token_usage"]
            meta_parts.append(
                f"🪙 {t['total_tokens']} tokens "
                f"(prompt {t['prompt_tokens']} + completion {t['completion_tokens']}) · "
                f"est. cost ${t['estimated_cost_usd']:.5f}"
            )
        st.caption("  ·  ".join(meta_parts))

        # Citation validity warning
        if not result["citations_valid"]:
            st.warning(
                "⚠️ Some citation indices in the answer don't match the sources below. "
                "This can happen in mock mode."
            )

        st.divider()

        # Source cards
        st.subheader("Sources")
        sources = result.get("sources", [])
        if sources:
            for src in sources:
                with st.expander(
                    f"[{src['citation_index']}] **{src['method']}** "
                    f"({src.get('year', '')}) — "
                    f"{src['file_name']}, page {src['page_number']}  "
                    f"· score {src['score']:.3f}"
                ):
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.markdown(f"**Method:** {src.get('method', 'N/A')}")
                        st.markdown(f"**Year:** {src.get('year', 'N/A')}")
                        st.markdown(f"**Venue:** {src.get('venue', 'N/A')}")
                        st.markdown(f"**Page:** {src.get('page_number', 'N/A')} / {src.get('page_count', '?')}")
                        st.markdown(f"**Chunk ID:** `{src.get('chunk_id', 'N/A')}`")
                    with col2:
                        st.markdown("**Excerpt:**")
                        text = src.get("text", "")
                        st.text(text[:500] + ("…" if len(text) > 500 else ""))
        else:
            st.info("No sources returned.")

        # Raw chunks (debug)
        if show_raw and result.get("raw_chunks"):
            st.divider()
            st.subheader("Raw retrieved chunks (debug)")
            for i, chunk in enumerate(result["raw_chunks"]):
                with st.expander(f"Chunk {i + 1} — {chunk.get('chunk_id', '')}"):
                    st.json({k: v for k, v in chunk.items() if k != "embedding"})

        # Feedback
        st.divider()
        st.caption("Was this answer helpful?")
        col_up, col_down, _ = st.columns([1, 1, 8])
        if col_up.button("👍", key="up"):
            pipeline.log_feedback(query=query, answer=result["answer"], helpful=True)
            st.success("Thanks!")
        if col_down.button("👎", key="down"):
            pipeline.log_feedback(query=query, answer=result["answer"], helpful=False)
            st.info("Noted — will help improve the system.")

    elif ask_clicked and not query.strip():
        st.warning("Please enter a question first.")

    # Query log viewer (collapsed by default)
    with st.expander("📋 Recent query log", expanded=False):
        log_path = Path("logs/query_log.jsonl")
        if log_path.exists():
            lines = log_path.read_text().splitlines()
            records = [json.loads(l) for l in lines if l.strip()][-10:]
            if records:
                log_df = pd.DataFrame([{
                    "Time":     r.get("timestamp", "")[-8:],
                    "Question": r.get("question", "")[:50],
                    "Tokens":   r.get("token_usage", {}).get("total_tokens", 0),
                    "ms":       r.get("latency_ms", 0),
                } for r in reversed(records)])
                st.dataframe(log_df, use_container_width=True)
            else:
                st.info("No queries logged yet.")
        else:
            st.info("No query log found — run a query first.")


if __name__ == "__main__":
    main()