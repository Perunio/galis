import streamlit as st
import pandas as pd
import os
import time
import random
import torch
import json
from streamlit.components.v1 import html
from ogb.nodeproppred import PygNodePropPredDataset # type: ignore


def copy_button_component(text_to_copy: str, button_text: str = "Copy to Clipboard"):
    """Creates a HTML/JS component with copy button."""
    escaped_text = json.dumps(text_to_copy)

    component = f"""
    <style>
        .custom-copy-button {{
            background-color: #4A4A70; color: white; border: none; padding: 10px 20px;
            text-align: center; font-size: 16px; cursor: pointer; border-radius: 8px; width: 100%;
            font-weight: bold;
        }}
        .custom-copy-button:hover {{ background-color: #5A5AC8; }}
        .custom-copy-button:active {{ background-color: #3E3E6B; }}
    </style>
    <button class="custom-copy-button" onclick="copyToClipboard(this)">{button_text}</button>
    <script>
        function copyToClipboard(element) {{
            navigator.clipboard.writeText({escaped_text}).then(function() {{
                const originalText = element.innerText;
                element.innerText = 'Copied!';
                setTimeout(() => {{ element.innerText = originalText; }}, 2000);
            }}, function(err) {{
                console.error('Could not copy text: ', err);
                alert('Failed to copy text.');
            }});
        }}
    </script>
    """
    html(component, height=50)


@st.cache_data(show_spinner="Downloading OGBN-Arxiv dataset...")
def load_data():
    large_sample_abstracts = [
        "This paper explores graph neural networks for large-scale citation data.",
        "A novel approach to optimizing transformer models using low-rank factorization.",
        "We study the impact of regularization techniques on node classification.",
        "This work presents a new dataset for benchmarking graph algorithms.",
        "An analysis of attention mechanisms in large language models.",
        "Investigating the robustness of convolutional neural networks to adversarial attacks.",
        "A deep learning framework for anomaly detection in time-series data.",
        "Exploring few-shot learning for image classification tasks.",
        "The development of a novel reinforcement learning agent for complex games.",
        "A comparative study of self-supervised learning methods for computer vision.",
        "Graph-based semi-supervised learning for text categorization.",
        "Optimizing distributed training of large-scale neural networks.",
        "A new method for natural language generation with stylistic control.",
        "Predicting protein structures using equivariant graph neural networks.",
        "Bayesian optimization for hyperparameter tuning in machine learning models.",
    ]

    dataset = PygNodePropPredDataset(name="ogbn-arxiv", root="./data")
    data = dataset[0]
    node_labels_df = pd.DataFrame(
        {
            "node_id": torch.arange(data.num_nodes).numpy(),
            "label_id": data.y.squeeze().numpy(),
        }
    )
    mapping_dir = os.path.join(dataset.root, "mapping")
    node_ids_df = pd.read_csv(
        os.path.join(mapping_dir, "nodeidx2paperid.csv.gz"),
        compression="gzip",
        header=None,
        names=["node_id", "paper_id"],
        dtype={"node_id": int, "paper_id": str},
        skiprows=1,
    )
    label_names_df = pd.read_csv(
        os.path.join(mapping_dir, "labelidx2arxivcategeory.csv.gz"),
        compression="gzip",
        header=None,
        names=["label_id", "category_name"],
        skiprows=1,
    )
    merged_df = pd.merge(node_ids_df, node_labels_df, on="node_id")
    full_df = pd.merge(merged_df, label_names_df, on="label_id")

    full_df["abstract"] = [
        random.choice(large_sample_abstracts) for _ in range(len(full_df))
    ]

    return full_df


def demo():
    st.set_page_config(page_title="Related Work Generator", layout="wide")

    st.markdown(
        """
    <style>
        .block-container { padding-top: 2rem; padding-bottom: 2rem; }
        .stApp { background-color: #0E1117; }
        h2 { color: #FFFFFF; font-weight: bold; }
        .stTextArea textarea {
            background-color: #1E1E1E; border: 1px solid #D32F2F; color: #FFFFFF;
            min-height: 250px; border-radius: 8px;
        }
        .stFileUploader {
            background-color: #262730; border: 1px solid #444; border-radius: 8px; padding: 15px;
        }
        .stFileUploader p { color: #FAFAFA; }
        div.stButton > button:not(.custom-copy-button) {
             background-color: #444; border-color: #555;
        }
        .generated-box {
            border: 1px solid #4F4F8C; padding: 20px; border-radius: 10px;
            background-color: #1E1E1E; min-height: 400px; color: #FFFFFF;
        }
        .generated-box ul { list-style-type: none; padding-left: 0; }
        .generated-box li { margin-bottom: 1.5em; line-height: 1.6; }
        .generated-box code { color: #81C784; background-color: transparent; }
    </style>
    """,
        unsafe_allow_html=True,
    )

    if "generated_text_md" not in st.session_state:
        st.session_state.generated_text_md = ""
    if "raw_text_for_copy" not in st.session_state:
        st.session_state.raw_text_for_copy = ""

    papers_data = load_data()
    st.title("Galis")
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.header("Abstract")
        abstract_input = st.text_area(
            "",
            key="abstract_text",
            height=250,
            placeholder="Paste your abstract here",
            label_visibility="collapsed",
        )
        st.write("Write or **Upload** Abstract Paper section.")
        uploaded_file = st.file_uploader(
            "Drag and drop file here", type=["txt"], help="Limit 200MB per file • TXT"
        )
        st.write("")

        if st.button("Generate Related Work", use_container_width=True):
            with st.spinner("Fetching random records..."):
                time.sleep(0.5)
                random_records = papers_data.sample(n=5)

                if not random_records.empty:
                    html_items = []
                    raw_text_parts = []

                    for _, row in random_records.iterrows():
                        category_str = (
                            f"arxiv {row['category_name'].lower().replace('-', ' ')}"
                        )

                        html_items.append(
                            f"<li>Paper ID:<br>{row['paper_id']}<br><br>Category:<br>{category_str}<br><br>{row['abstract']}</li>"
                        )

                        raw_text_parts.append(
                            f"Paper ID: {row['paper_id']}\nCategory: {category_str}\n\n{row['abstract']}"
                        )

                    st.session_state.generated_text_md = (
                        f"<ul>{''.join(html_items)}</ul>"
                    )
                    st.session_state.raw_text_for_copy = "\n\n---\n\n".join(
                        raw_text_parts
                    )
                else:
                    st.session_state.generated_text_md = (
                        "Could not fetch random records."
                    )
                    st.session_state.raw_text_for_copy = ""

    with col2:
        st.header("References")

        content = (
            st.session_state.generated_text_md
            if st.session_state.generated_text_md
            else " "
        )
        html_box = f"<div class='generated-box'>{content}</div>"
        st.markdown(html_box, unsafe_allow_html=True)

        if st.session_state.raw_text_for_copy:
            st.write("")
            copy_button_component(
                st.session_state.raw_text_for_copy, "Copy All Related Work"
            )


if __name__ == "__main__":
    demo()
