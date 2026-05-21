import streamlit as st
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Research Paper Recommendation System",
    page_icon="📚",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>
.stApp { background-color: #F5F7FA; }

.stTextInput > div > div > input {
    background-color: white;
    color: #202124;
    border-radius: 12px;
    border: 1px solid #DADCE0;
    padding: 14px;
    font-size: 16px;
}

.stButton > button {
    background-color: #43A047;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    border: none;
    font-size: 18px;
    font-weight: 600;
    transition: 0.3s;
}

.stButton > button:hover {
    background-color: #2E7D32;
    color: white;
}

section[data-testid="stSidebar"] { background-color: #E8F0FE; }
</style>
""", unsafe_allow_html=True)


# =========================================================
# CACHED LOADERS
# =========================================================

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_data
def load_data():
    return pd.read_csv("data/cleaned_data.csv")

@st.cache_resource
def load_faiss_index():
    return faiss.read_index("models/faiss_index.index")


# =========================================================
# INITIALIZE
# =========================================================

model = load_model()
df    = load_data()
index = load_faiss_index()


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("About Project")
st.sidebar.markdown("""
This project recommends research papers using:
- **BERT Embeddings**
- **FAISS Similarity Search**
- **Semantic NLP**
- **Deep Learning**

### Technologies Used
- Python · Streamlit
- Sentence Transformers · FAISS · Scikit-learn
""")

top_k = st.sidebar.slider("Number of Recommendations", min_value=1, max_value=10, value=5)

st.sidebar.markdown("---")
st.sidebar.success("Frontend integrated with BERT + FAISS backend.")


# =========================================================
# MAIN UI
# =========================================================

st.title("📚 Research Paper Recommendation System")
st.subheader("Semantic Search using BERT + FAISS")
st.write("")

query = st.text_input("Enter a research topic or keywords")


# =========================================================
# SEARCH & DISPLAY
# =========================================================

if st.button("Get Recommendations"):

    if not query.strip():
        st.warning("Please enter a research topic.")

    else:
        with st.spinner("Generating recommendations..."):

            # Embed query
            query_embedding = model.encode(
                [query], normalize_embeddings=True
            )

            # FAISS search
            distances, indices = index.search(
                np.array(query_embedding).astype("float32"),
                top_k
            )

            # ---------------------------------------------------
            # Sort by highest similarity (descending)
            # FAISS with IndexFlatIP returns dot-product scores;
            # higher = more similar. Zip, filter, then sort.
            # ---------------------------------------------------
            results = [
                (float(dist), int(idx))
                for dist, idx in zip(distances[0], indices[0])
                if idx != -1          # skip empty FAISS slots
            ]

            results.sort(key=lambda x: x[0], reverse=True)   # highest first

        st.success(f"✅ Top {len(results)} recommendations — most relevant first!")
        st.write("")

        # --------------------------------------------------
        # Display each recommended paper
        # --------------------------------------------------
        for rank, (dist, idx) in enumerate(results, start=1):

            row = df.iloc[idx]

            # Cosine similarity → percentage
            similarity = dist * 100

            # Badge colour: green → yellow → red as rank increases
            if rank == 1:
                badge = "🥇"
            elif rank == 2:
                badge = "🥈"
            elif rank == 3:
                badge = "🥉"
            else:
                badge = f"#{rank}"

            with st.container():
                st.markdown(f"### {badge} {row.get('title', 'Untitled')}")

                col1, col2 = st.columns([3, 1])

                with col1:
                    # Authors
                    if "authors" in row and pd.notna(row.get("authors")):
                        st.caption(f"✍️  {row['authors']}")

                    # Abstract / summary (truncated)
                    summary = (
                        row.get("abstract")
                        or row.get("summary")
                        or row.get("description")
                        or ""
                    )
                    if summary:
                        st.write(summary[:500] + ("..." if len(summary) > 500 else ""))

                with col2:
                    st.metric("Similarity", f"{similarity:.1f}%")

                    if "year" in row and pd.notna(row.get("year")):
                        st.write(f"📅 {int(row['year'])}")

                    if "venue" in row and pd.notna(row.get("venue")):
                        st.write(f"🏛️ {row['venue']}")

                st.divider()


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    "<div style='text-align:center; color:#777; margin-top:50px; font-size:14px;'>"
    "Built with Streamlit · BERT Embeddings · FAISS Semantic Search"
    "</div>",
    unsafe_allow_html=True
)