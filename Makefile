.PHONY: init lint test train eval dvc-init dvc-track dvc-push install-hooks validate-data

PY=python

init:
	python -m pip install -U pip
	pip install -r requirements.txt

install-hooks:
	@echo "Installing Git hooks..."
	@if [ -f ".github/hooks/install-hooks.sh" ]; then \
		bash .github/hooks/install-hooks.sh; \
	else \
		echo "Error: install-hooks.sh not found"; \
		exit 1; \
	fi

lint:
	ruff check .
	black --check .

test:
	pytest -q

train:
	HF_LOCAL_ONLY=1 HF_CACHE_DIR=$$(pwd)/models/roberta-base DATA_ROOT=$$(pwd)/data/processed \
	$(PY) scripts/train.py --profile local

eval:
	HF_LOCAL_ONLY=1 HF_CACHE_DIR=$$(pwd)/models/roberta-base DATA_ROOT=$$(pwd)/data/processed \
	$(PY) scripts/train.py --profile server --eval-only

dvc-init:
	dvc init -q
	dvc remote add -d local ./dvcstore || true

dvc-track:
	dvc add data/processed || true
	git add data/processed.dvc .gitignore

dvc-push:
	dvc push

validate-data:
	$(PY) scripts/validate_data_schema.py
