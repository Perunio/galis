FROM python:3.12.4-slim

WORKDIR /app

RUN pip install poetry

COPY pyproject.toml ./
RUN poetry install --no-root
RUN poetry run pip install torch-scatter torch-sparse torch-cluster pyg-lib -f https://data.pyg.org/whl/torch-2.3.1+cu121.html
RUN poetry run pip install torch-geometric

COPY galis_app.py ./
COPY model ./model
COPY dataset ./dataset
COPY predictor ./predictor
COPY llm ./llm

ENV GOOGLE_API_KEY=""

EXPOSE 7860

CMD ["poetry", "run", "streamlit", "run", "galis_app.py", "--server.port=7860", "--server.address=0.0.0.0"]
