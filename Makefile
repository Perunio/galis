install:
	poetry install --no-root
	poetry run pip install torch-scatter torch-sparse torch-cluster pyg-lib -f https://data.pyg.org/whl/torch-2.3.1+cu121.html
	poetry run pip install torch-geometric

format:
	poetry run ruff format .
	poetry run ruff check . --fix

check:
	poetry run ruff format --check .

test:
	poetry run pytest ./tests -vv

demo:
	poetry run streamlit run ./demo.py

run-app:
	poetry run streamlit run ./galis_app.py