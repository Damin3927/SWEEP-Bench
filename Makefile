.PHONY: lint lint-fix format format-check typecheck check

lint:
	uv run --no-sync ruff check .

lint-fix:
	uv run --no-sync ruff check --fix .

format:
	uv run --no-sync ruff format .

format-check:
	uv run --no-sync ruff format --check .

typecheck:
	uv run --no-sync ty check sweep_bench scripts

check: lint format-check typecheck
