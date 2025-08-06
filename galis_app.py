import sys
from pathlib import Path
import streamlit as st

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from predictor.link_predictor import (
    prepare_system,
    get_citation_predictions,
    abstract_to_vector,
    format_top_k_predictions,
)

MODEL_PATH = Path("predictor/model.pth")


@st.cache_resource
def load_prediction_system(model_path):
    return prepare_system(model_path)


def app():
    st.set_page_config(page_title="Galis", layout="wide")
    st.title("Galis")

    gcn_model, st_model, dataset, z_all = load_prediction_system(MODEL_PATH)

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.header("Abstract Title")
        if "abstract_title" not in st.session_state:
            st.session_state.abstract_title = ""
        abstract_title = st.text_input(
            "Paste your title here",
            key="abstract_title_input",
            label_visibility="collapsed",
        )

        st.header("Abstract Text")
        if "abstract_text" not in st.session_state:
            st.session_state.abstract_text = ""
        abstract_input = st.text_area(
            "Paste your abstract here",
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

        if st.button("Suggest References", type="primary"):
            if not abstract_title.strip() or not abstract_input.strip():
                st.warning("Please provide both a title and an abstract.")
            else:
                with col2:
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

    with col2:
        st.header("Suggested References")
        if "references" in st.session_state:
            st.markdown(f"```\n{st.session_state.references}\n```")


if __name__ == "__main__":
    app()
