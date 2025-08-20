from pathlib import Path
import streamlit as st

from predictor.link_predictor import (
    prepare_system,
    get_citation_predictions,
    abstract_to_vector,
    format_top_k_predictions,
)
from llm.related_work_generator import generate_related_work

MODEL_PATH = Path("model.pth")


@st.cache_resource
def load_prediction_system(model_path):
    return prepare_system(model_path)


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

    gcn_model, st_model, dataset, z_all = load_prediction_system(MODEL_PATH)

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
                    new_vector = abstract_to_vector(
                        abstract_input, abstract_title, st_model
                    )
                    probabilities = get_citation_predictions(
                        vector=new_vector,
                        model=gcn_model,
                        z_all=z_all,
                        num_nodes=dataset.data.num_nodes,
                    )
                    references = format_top_k_predictions(
                        probabilities, dataset, top_k=num_citations
                    )
                    st.session_state.references = references

                with references_placeholder.container():
                    st.header("Suggested References")
                    with st.container(height=200):
                        st.markdown(st.session_state.references)

                with related_work_placeholder.container():
                    with st.spinner("Generating related work section..."):
                        related_work = generate_related_work(
                            st.session_state.references
                        )
                        st.session_state.related_work = related_work

    if st.session_state.references:
        with references_placeholder.container():
            st.header("Suggested References")
            with st.container(height=200):
                st.markdown(st.session_state.references)

    if st.session_state.related_work:
        with related_work_placeholder.container():
            st.header("Suggested Related Works")
            with st.container(height=200):
                st.markdown(st.session_state.related_work)


if __name__ == "__main__":
    app()
