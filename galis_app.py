from pathlib import Path
import streamlit as st
from dataset.ogbn_link_pred_dataset import OGBNLinkPredDataset
from model.paper_similarity import PaperSimilarityFinder
from llm.related_work_generator import generate_related_work


@st.cache_resource
def load_similarity_finder():
    dataset = OGBNLinkPredDataset()
    model_name = "all-mpnet-base-v2"
    embeddings_dir = Path("embeddings_cache")

    similarity_finder = PaperSimilarityFinder(
        dataset,
        method="sentence_transformer",
        model_name=model_name,
        embeddings_cache_path=embeddings_dir,
    )
    return similarity_finder, dataset


def format_top_k_predictions_from_similarity(similar_papers: list) -> str:
    markdown_list = []
    for i, (idx, score, text) in enumerate(similar_papers):
        title = text.split('\n')[0].strip()
        markdown_list.append(f"{i + 1}. **{title}** (Similarity: {score:.4f})")
    return "\n".join(markdown_list)


def app():
    st.set_page_config(page_title="Galis", layout="wide")
    st.title("Galis")

    if "references" not in st.session_state:
        st.session_state.references = None
    if "related_work" not in st.session_state:
        st.session_state.related_work = None
    if "abstract_title" not in st.session_state:
        st.session_state.abstract_title = ""
    if "abstract_text" not in st.session_state:
        st.session_state.abstract_text = ""

    similarity_finder, dataset = load_similarity_finder()

    col1, col2 = st.columns(2, gap="large")

    with col2:
        references_placeholder = st.empty()
        related_work_placeholder = st.empty()

    with col1:
        st.header("Abstract Title")
        abstract_title = st.text_input(
            "Paste your title here",
            st.session_state.abstract_title,
            key="abstract_title_input",
            label_visibility="collapsed",
        )

        st.header("Abstract Text")
        abstract_input = st.text_area(
            "Paste your abstract here",
            st.session_state.abstract_text,
            key="abstract_text_input",
            height=100,
            label_visibility="collapsed",
        )

        st.write("...or **upload** a .txt file (first line = title, rest = abstract)")
        uploaded_file = st.file_uploader(
            "Drag and drop file here", type=["txt"], help="Limit 200MB per file • TXT"
        )

        if uploaded_file is not None:
            content = uploaded_file.getvalue().decode("utf-8").splitlines()
            st.session_state.abstract_title = content[0] if content else ""
            st.session_state.abstract_text = (
                "\n".join(content[1:]) if len(content) > 1 else ""
            )
            st.rerun()

        st.session_state.abstract_title = abstract_title
        st.session_state.abstract_text = abstract_input

        num_citations = st.number_input(
            "Number of suggestions",
            min_value=1,
            max_value=100,
            value=10,
            step=1,
            help="Choose how many paper suggestions you want to see.",
        )

        if st.button("Suggest References and related work", type="primary"):
            if not abstract_title.strip() or not abstract_input.strip():
                st.warning("Please provide both a title and an abstract.")
            else:
                st.session_state.references = None
                st.session_state.related_work = None
                references_placeholder.empty()
                related_work_placeholder.empty()

                with st.spinner("Analyzing abstract and predicting references..."):
                    similar_papers = similarity_finder.find_similar_papers(
                        title=abstract_title,
                        abstract=abstract_input,
                        top_k=num_citations
                    )
                    references = format_top_k_predictions_from_similarity(similar_papers)
                    st.session_state.references = references

                with references_placeholder.container():
                    st.header("Suggested References")
                    with st.container(height=200):
                        st.markdown(st.session_state.references, unsafe_allow_html=True)

                with related_work_placeholder.container():
                    with st.spinner("Generating related work section..."):
                        related_work = generate_related_work(
                            st.session_state.abstract_title,
                            st.session_state.abstract_text,
                            st.session_state.references,
                        )
                        st.session_state.related_work = related_work

    if st.session_state.references:
        with references_placeholder.container():
            st.header("Suggested References")
            with st.container(height=200):
                st.markdown(st.session_state.references, unsafe_allow_html=True)

    if st.session_state.related_work:
        with related_work_placeholder.container():
            st.header("Suggested Related Works")
            with st.container(height=200):
                st.markdown(st.session_state.related_work, unsafe_allow_html=True)


if __name__ == "__main__":
    app()