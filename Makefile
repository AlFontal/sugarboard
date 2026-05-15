.PHONY: run lint format test test-e2e test-e2e-baseline ci coverage install prod-install clean

run: install
	NICEGUI_RELOAD=true ./venv/bin/python nicegui_app.py

lint: install
	./venv/bin/ruff check .
	./venv/bin/ruff format --check .

format: install
	./venv/bin/ruff check --fix .
	./venv/bin/ruff format .

test: install
	./venv/bin/python -m pytest -ra -v -m "not e2e" --cov-report=html:coverage --cov-config=pyproject.toml --cov-report=term-missing --cov=. ./tests

test-e2e: install
	./venv/bin/python -m pytest -ra -v -m e2e ./tests

test-e2e-baseline: install
	./venv/bin/python -m pytest -ra -v -m e2e --visual-baseline ./tests

ci: lint test

coverage: install
	./venv/bin/python -m http.server --bind 127.0.0.1 --directory coverage

install: requirements.dev.txt
	[ -d venv ] || python -m venv venv
	./venv/bin/pip install -r requirements.dev.txt

prod-install: requirements.txt
	python -m pip install -r requirements.txt

clean:
	rm -rf __pycache__ .pytest_cache .ruff_cache coverage htmlcov .coverage
