.PHONY: setup run check

setup:
	python3 -m venv .venv
	. .venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt

run:
	python gesture2/main.py

check:
	python -m py_compile $$(find gesture2 -name '*.py')
