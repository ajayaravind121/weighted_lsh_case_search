import streamlit as st
from similarity_engine import WeightedLSH

st.set_page_config(page_title="Medical Similar Case Search", layout="wide")
st.title("🩺 Medical Similar Case Search Engine (Weighted LSH)")
st.write("Enter a medical note below to find similar cases from the dataset.")

@st.cache_resource
def load_engine():
    engine = WeightedLSH(
        csv_path="data/mtsamples.csv",
        max_docs=1500,
        num_hashes=100,
        b=20,
        r=5,
    )
    engine.build()
    return engine

engine = load_engine()

col_input, col_results = st.columns([1, 2])

with col_input:
    st.subheader("Input Medical Note")
    example = (
        "Patient presents with chest pain radiating to the left arm, "
        "shortness of breath, and sweating."
    )
    query_text = st.text_area("Enter a clinical note:", example, height=200)
    top_k = st.slider("Number of similar cases:", 3, 20, 5)

    search_clicked = st.button("Search")

    if search_clicked:
        if not query_text.strip():
            st.warning("Please enter text.")
        else:
            with st.spinner("Finding similar cases..."):
                results = engine.query(query_text, topk=top_k)
            st.session_state["results"] = results

with col_results:
    st.subheader("Similar Cases")
    results = st.session_state.get("results", None)

    if not results:
        st.info("Results will appear here.")
    else:
        scores = [r["similarity"] for r in results]
        avg = sum(scores) / len(scores) if scores else 0.0
        st.write(f"**Average similarity score of top {len(results)} cases:** `{avg:.3f}`")

        for i, r in enumerate(results, start=1):
            # Use Streamlit's Markdown color syntax
            header = (
                f"#{i} – Similarity: :green[{r['similarity']:.3f}] "
                f"– Specialty: :violet[{r['medical_specialty']}]"
            )
            with st.expander(header):
                st.markdown(f"**Description:** {r['description']}")
                st.markdown(f"**Sample name:** {r['sample_name']}")
                st.markdown(f"**Keywords:** {r['keywords']}")
                st.markdown("**Transcription:**")
                st.write(r["transcription"])
