from fastapi import FastAPI, HTTPException, status
import uvicorn
import os
from dotenv import load_dotenv
from pydantic import BaseModel
import logging
import httpx
from pipeline import Pipeline
from datetime import datetime

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app_port = int(os.getenv("ML_SERVICE_PORT"))
app_host = os.getenv("ML_SERVICE_HOST")

db_host = os.getenv("DB_SERVICE_HOST")
db_port = os.getenv("DB_SERVICE_PORT")
db_service_url = f"http://{db_host}:{db_port}"

app = FastAPI(title="ML Service")

@app.get("/")
def index():
    return {"message": "ML Service API"}

@app.post(
    path="/start-scrap",
    name="start scraping from google scholar",
    status_code=status.HTTP_200_OK
)
async def start_scraping():
    try:
        pipeline = Pipeline(query='machine learning', max_results=20)
        logger.info(f"Starting pipeline with query: {pipeline.query}, max results: {pipeline.max_results}")
        results = pipeline.run_pipeline()

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{db_service_url}/save-documents",
                    json=results
                )

                response.raise_for_status()
            except Exception as e:   
                logger.error(f"Error saving result to DB: {str(e)}")

        return {"message": "Scraping successfully executed", "count":{len(results)}}
    except Exception as e:
        logger.error(f"Error during scraping: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occured while running the scraping pipeline"
        )

@app.get("/health")
async def health_check():
    try:
        async with httpx.AsyncClient() as client:
            db_response = await client.get(f"{db_service_url}/health", timeout=2.0)
            db_status = "ok" if db_response.status_code == 200 else "error"
        
        return {
            "status": "healthy" if db_status == "ok" else "degraded",
            "db_service_connection": db_status,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

if __name__ == "__main__":
    uvicorn.run(
        app=app,
        host=app_host,
        port=app_port
    )        