install-dependecies:
	pip install -r requirements.txt
	python -c "import nltk; nltk.download('punkt_tab')"

ml-run:
	python src/ml/app.py

backend-run:
	python src/backend/app/main.py

db-run:
	python src/db/app.py