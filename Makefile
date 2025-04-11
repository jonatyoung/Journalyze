ml-run:
	python src/ml/app.py

backend-run:
	python src/backend/app/main.py

db-run:
	python src/db/app.py

compose-up:
	docker-compose up -d --build

compose-down:
	docker-compose down	