.PHONY: run run-elevenlabs run-gemini voices models install venv clean

VENV_DIR := .venv
PYTHON := $(VENV_DIR)/bin/python
PIP := $(VENV_DIR)/bin/pip

run:
	@if [ ! -d "$(VENV_DIR)" ]; then \
		echo "Virtual environment not found. Please run 'make install' first."; \
		exit 1; \
	fi
	$(PYTHON) app.py --provider elevenlabs --model eleven_v4

run-elevenlabs: run

run-gemini:
	$(PYTHON) app.py --provider gemini

voices:
	$(PYTHON) app.py --list-voices

models:
	$(PYTHON) app.py --list-models

install: venv
	$(PIP) install -r requirements.txt

venv:
	python3 -m venv $(VENV_DIR)

clean:
	rm -rf $(VENV_DIR)
	rm -f output.wav
	find . -type d -name "__pycache__" -exec rm -rf {} +
