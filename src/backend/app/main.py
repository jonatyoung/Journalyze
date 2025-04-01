from fastapi import FastAPI, status, HTTPException
import uvicorn
from bertopic import BERTopic
import pickle
import os
import json
import logging
from pathlib import Path
from dotenv import load_dotenv
from src.ml.pipeline import Pipeline
from src.db.mongo import connect_to_mongodb
from src.backend.app.scrap_result_model import ScrapingResult, mongo_to_pydantic

MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../models/bertopic_model.pkl"))

load_dotenv()

app_port = int(os.getenv("APP_PORT"))
app_host = os.getenv("APP_HOST")

app = FastAPI()

mongo_collection = connect_to_mongodb()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print(f'run the model from the path: {MODEL_PATH}')

try:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"model file not found in: {MODEL_PATH}")

    with open(MODEL_PATH, "rb") as model_file:
        model: BERTopic = pickle.load(model_file)
except Exception as e:
    print(f"error : {e}")    

@app.get('/')
def index():
    return {'message': 'Topic Modeling API'}

@app.post(
    path="/start-scrap",
    name="start scraping from google scholar",
    status_code=status.HTTP_200_OK
)
def start_scraping():
    pipeline = Pipeline(query='machine learning', max_results=20)

    try:
        pipeline.run_pipeline()
        return {"message": "Scraping successfully executed"}
    except Exception as e:
        logger.error(f"Error during scraping: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occured while running the scraping pipeline"
        )

@app.get(
    '/scraped-results',
    response_model=list[ScrapingResult])
def get_scraped_result():
    try:
        results = list(mongo_collection.find().limit(10))

        if not results:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="The result is empty"
            )

        logger.info(f"Fetched {len(results)} results from MongoDB.")
        
        return [mongo_to_pydantic(doc) for doc in results]
    except Exception as e:
        logger.error(f"Error fetching data: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error fetching data")

if __name__ == '__main__':
    uvicorn.run(
        app=app,
        host=app_host,
        port=app_port
    )