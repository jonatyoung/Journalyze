FROM pytorch/pytorch:latest

WORKDIR /app

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN python -c "import nltk; nltk.download('punkt_tab')"

COPY . .

EXPOSE 8001

CMD [ "python", "app.py" ]