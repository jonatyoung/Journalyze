from fastapi import FastAPI
import uvicorn
from bertopic import BERTopic
import pickle
import os
import json
from pathlib import Path
from dotenv import load_dotenv

MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../models/bertopic_model.pkl"))

load_dotenv()
app_port:int = int(os.getenv("APP_PORT"))

print(f'run the model from the path: {MODEL_PATH}')

try:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"model file not found in: {MODEL_PATH}")

    with open(MODEL_PATH, "rb") as model_file:
        model: BERTopic = pickle.load(model_file)
except Exception as e:
    print(f"error : {e}")    

app = FastAPI()

@app.get('/')
def index():
    return {'message': 'Topic Modeling API'}

@app.get('/journal')
def get_journals_with_topics():
    topic_info = model.get_topic_info()

    topic_info_json = topic_info.to_json(orient="records", force_ascii=False)

    return {'results' : topic_info_json}

if __name__ == '__main__':
    uvicorn.run(app=app, host='127.0.0.1', port=app_port)