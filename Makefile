# Task shortcuts. On Windows run these from Git Bash (or call the commands directly).
PY ?= .venv/Scripts/python
ifeq ($(OS),)
PY = .venv/bin/python
endif

.PHONY: install demo serve test lint format benchmark export-onnx sample docker

install:            ## Install runtime + dev dependencies into the active venv
	$(PY) -m pip install -r requirements-dev.txt && $(PY) -m pip install -e . --no-deps

demo:               ## Download the demo video (CC BY 4.0, Intel sample-videos)
	$(PY) scripts/download_demo.py

serve: demo         ## Run API + dashboard on http://127.0.0.1:8103
	$(PY) -m rva.cli serve

test:               ## Run the test suite
	$(PY) -m pytest -q

lint:               ## Ruff lint + format check
	$(PY) -m ruff check . && $(PY) -m ruff format --check .

format:
	$(PY) -m ruff check --fix . && $(PY) -m ruff format .

export-onnx:        ## Export yolo11n to ONNX (640 and 320)
	$(PY) -m rva.cli export-onnx --imgsz 640 && $(PY) -m rva.cli export-onnx --imgsz 320

benchmark: demo     ## CPU benchmark PyTorch vs ONNX
	$(PY) scripts/benchmark.py --frames 150 --sizes 320 640

sample: demo        ## Save an annotated sample frame to docs/sample_output.jpg
	$(PY) -m rva.cli process --max-frames 400 --frame-out docs/sample_output.jpg --frame-at 400

docker:
	docker compose up --build
