install-dependecies:
	pip install -r requirements.txt
	python -c "import nltk; nltk.download('punkt_tab')"

scrap-run:
	python -m src.ml.pipeline

backend-run:
	python -m src.backend.app.main

train-model:
	python src/ml/train.py	