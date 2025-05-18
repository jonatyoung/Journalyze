import uvicorn
import os
from dotenv import load_dotenv
import logging
import httpx
import nltk
import json
from pipeline import Pipeline
from datetime import datetime
from prometheus_client import Summary, start_http_server
from mlflow_train import MLflowTrainer
from typing import Dict, Any, Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, HTTPException, status, Query

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

prometheus_client_port = int(os.getenv("PROMETHEUS_CLIENT_PORT", "8001"))
app_port = int(os.getenv("ML_SERVICE_PORT", "8000"))
app_host = os.getenv("ML_SERVICE_HOST", "0.0.0.0")
db_host = os.getenv("DB_SERVICE_HOST")
db_port = os.getenv("DB_SERVICE_PORT")
db_service_url = f"http://{db_host}:{db_port}"
mlhost = os.getenv("MLFLOW_HOST")

logger.info(f"host: {mlhost}")

# Metrics
SCRAPE_DURATION = Summary('scrape_duration_seconds', 'Time spent scraping')

app = FastAPI(title="ML Service")

mlflow_trainer = MLflowTrainer()

@app.get("/")
def index():
    return {"message": "ML Service API"}

@app.post(
    path="/train",
    name="train bertopic model",
    status_code=status.HTTP_200_OK
)
async def train_model(
    n_neighbors: int = Query(default=15, description="Number of neighbors parameter for UMAP"),
    n_components: int = Query(default=10, description="Number of components parameter for UMAP"),
    min_cluster_size: int = Query(default=25, description="Minimum cluster size for HDBSCAN")
):
    try:
        params = {
            "n_neighbors": n_neighbors,
            "n_components": n_components,
            "min_cluster_size": min_cluster_size
        }
        logger.info(f"Starting model training with parameters: {params}")
        
        with ThreadPoolExecutor() as executor:
            metrics = await asyncio.get_event_loop().run_in_executor(
                executor, 
                lambda: mlflow_trainer.train_model(params)
            )
        
        return {
            "message": "Model training completed successfully",
            "parameters": params,
            "metrics": metrics
        }
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error during training: {str(e)}"
        )

@app.post(
    path="/start-scrap",
    name="start scraping from google scholar",
    status_code=status.HTTP_200_OK
)
async def start_scraping(
    query: str = Query(default="computer vision", description="Search query for Google Scholar"), 
    max_results: int = Query(default=200, description="Maximum number of results to retrieve")
):
    try:
        with SCRAPE_DURATION.time():
            pipeline = Pipeline(query=query, max_results=max_results)
            logger.info(f"Starting pipeline with query: {pipeline.query}, max results: {pipeline.max_results}")
            results, _ = pipeline.run_pipeline()
            logger.info(f"Results to be sent: {json.dumps(results[:1], indent=2)}")

            async with httpx.AsyncClient() as client:
                try:
                    response = await client.post(
                        f"{db_service_url}/save-documents",
                        json=results
                    )
                    response.raise_for_status()
                except Exception as e:   
                    logger.error(f"Error saving result to DB: {str(e)}")
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail=f"Error saving results to database: {str(e)}"
                    )

        return {"message": "Scraping successfully executed", "count": len(results)}
    except Exception as e:
        logger.error(f"Error during scraping: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while running the scraping pipeline"
        )

@app.get("/health")
async def health_check():
    try:
        db_status = "ok"
        try:
            async with httpx.AsyncClient() as client:
                db_response = await client.get(f"{db_service_url}/health", timeout=2.0)
                if db_response.status_code != 200:
                    db_status = "error"
        except Exception:
            db_status = "error"
        
        mlflow_status = "ok" if mlflow_trainer.check_mlflow_connection() else "error"
        
        return {
            "status": "healthy" if (db_status == "ok" and mlflow_status == "ok") else "degraded",
            "db_service_connection": db_status,
            "mlflow_connection": mlflow_status,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

if __name__ == "__main__":
    start_http_server(prometheus_client_port)
    
    uvicorn.run(
        app=app,
        host=app_host,
        port=app_port
    )