from pathlib import Path
import streamlit as st
from dataset.ogbn_link_pred_dataset import OGBNLinkPredDataset
from model.paper_similarity import PaperSimilarityFinder
from llm.related_work_generator import (
    generate_related_work,
    create_related_work_pipeline,
)


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

    pipeline = create_related_work_pipeline()

    return pipeline, similarity_finder, dataset


def format_top_k_predictions_from_similarity(similar_papers: list) -> str:
    markdown_list = []
    for i, (idx, score, text) in enumerate(similar_papers):
        title = text.split("\n")[0].strip()
        markdown_list.append(f"{i + 1}. {title} (Similarity: {score:.4f})")
    return "\n".join(markdown_list)


def process_uploaded_file():
    try:
        uploaded_file = st.session_state.file_uploader
        if uploaded_file is not None:
            content = uploaded_file.getvalue().decode("utf-8").splitlines()
            st.session_state.abstract_title = content[0] if content else ""
            st.session_state.abstract_text = (
                "\n".join(content[1:]) if len(content) > 1 else ""
            )
    except Exception as e:
        st.error(f"Error processing file: {e}")


def app():
    st.set_page_config(page_title="Galis", layout="wide")
    st.title("Galis")
    with st.popover("What is Galis?"):
        st.markdown(
            """
        ### About GALIS

        **GALIS** is a web-based application designed to streamline and improve the creation of related work and 
        references sections for research papers. It leverages an existing semantic graph that captures the 
        relationships and core concepts among cited papers to guide language model outputs.
        
        ### Objective
        The primary objective is to provide a practical tool that helps researchers generate high-quality, coherent 
        related work and references sections, making the process of synthesizing literature more efficient and 
        insightful.
        
        ---
        
        ### How to Use GALIS
        
        #### Option 1: Manual Input
        1. **Enter your paper title** in the "Abstract Title" field
        2. **Paste your abstract** in the "Abstract Text" area
        3. **Set the number of suggestions** you want (1-100 papers)
        4. **Click "Suggest References and related work"**
        
        #### Option 2: File Upload
        1. **Prepare a .txt file** with:
           - **First line**: Your paper title
           - **Remaining lines**: Your abstract text
        2. **Upload the file** using the file uploader
        3. **Set the number of suggestions** you want (1-100 papers)
        4. **Click "Suggest References and related work"**
        
        #### What You'll Get
        - **Suggested References**: A curated list of relevant papers based on semantic similarity
        - **Related Work Section**: An automatically generated related work section that synthesizes the suggested 
        papers
        - **Regeneration Option**: You can regenerate the related work section if needed
        
        ---
        
        *Note: File uploads are limited to 200MB and must be in .txt format* 
        """
        )

    if "references" not in st.session_state:
        st.session_state.references = ""
    if "related_work" not in st.session_state:
        st.session_state.related_work = ""
    if "abstract_title" not in st.session_state:
        st.session_state.abstract_title = ""
    if "abstract_text" not in st.session_state:
        st.session_state.abstract_text = ""

    pipeline, similarity_finder, dataset = load_similarity_finder()

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.header("Abstract Title")
        st.text_input(
            "Paste your title here", key="abstract_title", label_visibility="collapsed"
        )

        st.header("Abstract Text")
        st.text_area(
            "Paste your abstract here",
            key="abstract_text",
            height=150,
            label_visibility="collapsed",
        )

        st.file_uploader(
            "Upload a .txt file here (first line = title, rest = abstract)",
            type=["txt"],
            help="Limit 200MB per file • TXT",
            key="file_uploader",
            on_change=process_uploaded_file,
        )

        num_citations = st.number_input(
            "Number of suggestions",
            min_value=1,
            max_value=100,
            value=10,
            step=1,
            help="Choose how many paper suggestions you want to see.",
        )

        if st.button("Suggest References and related work", type="primary"):
            if (
                not st.session_state.abstract_title.strip()
                or not st.session_state.abstract_text.strip()
            ):
                st.warning("Please provide both a title and an abstract.")
            else:
                st.session_state.references = "LOADING"
                st.session_state.related_work = ""

    with col2:
        if st.session_state.references == "LOADING":
            with st.spinner("Analyzing abstract and predicting references..."):
                similar_papers = similarity_finder.find_similar_papers(
                    title=st.session_state.abstract_title,
                    abstract=st.session_state.abstract_text,
                    top_k=num_citations,
                )
                st.session_state.references = format_top_k_predictions_from_similarity(
                    similar_papers
                )
                st.session_state.related_work = "LOADING"
                st.rerun()

        if st.session_state.references not in ["", "LOADING"]:
            st.header("Suggested References")
            st.text_area(
                "References",
                value=st.session_state.references,
                height=150,
                label_visibility="collapsed",
                key="ref_output",
            )

            st.header("Suggested Related Works")

            if st.session_state.related_work == "LOADING":
                with st.spinner("Generating related work section..."):
                    st.session_state.related_work = generate_related_work(
                        pipeline,
                        st.session_state.abstract_title,
                        st.session_state.abstract_text,
                        st.session_state.references,
                    )
                    st.rerun()
            else:
                st.text_area(
                    "Related Works",
                    value=st.session_state.related_work,
                    height=300,
                    label_visibility="collapsed",
                    key="rw_output",
                )

            if st.button("Regenerate Related Works"):
                st.session_state.related_work = "LOADING"
                st.rerun()


if __name__ == "__main__":
    app()
