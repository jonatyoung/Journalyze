from fastapi import FastAPI, HTTPException, status
import uvicorn
import os
import logging
from dotenv import load_dotenv
import httpx
from scrap_result_model import ScrapingResult
from datetime import datetime

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app_port = int(os.getenv("BE_SERVICE_PORT"))
app_host = os.getenv("BE_SERVICE_HOST")

logger.info(f"host: {app_host}, port: {app_port}")

ml_port = int(os.getenv("ML_SERVICE_PORT"))
ml_host = os.getenv("ML_SERVICE_HOST")
ml_service_url = f"http://{ml_host}:{ml_port}"

db_host = os.getenv("DB_SERVICE_HOST")
db_port = os.getenv("DB_SERVICE_PORT")
db_service_url = f"http://{db_host}:{db_port}"

app = FastAPI(title="Journalyze API Gateway")

@app.get("/")
def index():
    return {"message": "Welcome to Journalyze API"}

@app.post(
    path="/api/start-scrap",
    name="Start scraping from Google Scholar",
    status_code=status.HTTP_200_OK
)
async def start_scraping():
    try:
        logger.info(f"Starting scraping api")
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{ml_service_url}/start-scrap",
                timeout=60.0  # Increased timeout for longer operations
            )
            response.raise_for_status()
            
        return {"message": "Scraping successfully initiated"}
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error from ML service: {e.response.text}")
        raise HTTPException(
            status_code=e.response.status_code,
            detail=f"Error from ML service: {e.response.text}"
        )
    except Exception as e:
        logger.error(f"Error during scraping: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while running the scraping pipeline: {str(e)}"
        )

@app.get(
    path="/api/scraped-results",
    status_code=status.HTTP_200_OK,
    response_model=list[ScrapingResult]
)
async def get_scraped_results():
    try:
        logger.info(f"Starting fetch scraped results api")
        
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{db_service_url}/scraped-results"
            )
            response.raise_for_status()
            
            results = response.json()
            
            if not results:
                logger.warning("No scraped results found")
                return []
                
            return results
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error from DB service: {e.response.text}")
        raise HTTPException(
            status_code=e.response.status_code,
            detail=f"Error from DB service: {e.response.text}"
        )
    except Exception as e:
        logger.error(f"Error fetching scraped results: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching scraped results: {str(e)}"
        )

@app.get("/health")
async def health_check():
    health_status = {
        "status": "healthy",
        "components": {},
        "timestamp": datetime.now().isoformat()
    }
    
    # Check connection for ML Service
    try:
        async with httpx.AsyncClient() as client:
            ml_response = await client.get(
                f"{ml_service_url}/health", 
                timeout=2.0
            )
            if ml_response.status_code == 200:
                health_status["components"]["ml_service"] = {
                    "status": "ok",
                    "details": ml_response.json() if hasattr(ml_response, "json") else {}
                }
            else:
                health_status["components"]["ml_service"] = {
                    "status": "degraded",
                    "details": f"Status code: {ml_response.status_code}"
                }
                health_status["status"] = "degraded"
    except Exception as e:
        health_status["components"]["ml_service"] = {
            "status": "error",
            "details": str(e)
        }
        health_status["status"] = "degraded"
    
    # Check connection for DB Service
    try:
        async with httpx.AsyncClient() as client:
            db_response = await client.get(
                f"{db_service_url}/health", 
                timeout=2.0
            )
            if db_response.status_code == 200:
                health_status["components"]["db_service"] = {
                    "status": "ok",
                    "details": db_response.json() if hasattr(db_response, "json") else {}
                }
            else:
                health_status["components"]["db_service"] = {
                    "status": "degraded",
                    "details": f"Status code: {db_response.status_code}"
                }
                health_status["status"] = "degraded"
    except Exception as e:
        health_status["components"]["db_service"] = {
            "status": "error",
            "details": str(e)
        }
        health_status["status"] = "degraded"
    
    health_status["components"]["backend_service"] = {
        "status": "ok"
    }
    
    if any(component.get("status") == "error" for component in health_status["components"].values()):
        health_status["status"] = "unhealthy"
    
    return health_status

if __name__ == "__main__":
    uvicorn.run(
        app=app,
        host=app_host,
        port=app_port
    )