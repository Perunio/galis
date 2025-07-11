install:
	poetry install --no-root

format:
	poetry run ruff format .
	poetry run ruff check . --fix

check:
	poetry run ruff format --check .

test:
	poetry run pytest ./tests -vv

run-app:
	poetry run streamlit run ./run_app.py
