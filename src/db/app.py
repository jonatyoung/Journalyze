from mongo import connect_to_mongodb
from fastapi import FastAPI, HTTPException, status, Body
from scrap_result_model import ScrapingResult, mongo_to_pydantic
import uvicorn
import os
from dotenv import load_dotenv
import logging
from datetime import datetime

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app_port = int(os.getenv("DB_SERVICE_PORT"))
app_host = os.getenv("DB_SERVICE_HOST")

app = FastAPI(title="DB Service")

mongo_collection = connect_to_mongodb()

@app.get("/")
def index():
    return {"message": "DB Service API"}


@app.post(
    "/save-documents",
    summary="Save the scraped data set",
    status_code=status.HTTP_201_CREATED
)
def insert_documents(documents: list[dict] = Body(...)):
    try:
        insert_result = mongo_collection.insert_many(documents)
        inserted_count = len(insert_result.inserted_ids)
        logger.info(f"✅ Successfully saved {inserted_count} documents")
    except Exception as insert_error:
        logger.error(f"Error inserting documents: {insert_error}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error inserting documents: {insert_error}"
        )

@app.get(
    '/scraped-results',
    summary="Fetch a lot of scraped data from the database",
    # Hapus response_model agar tidak memaksa konversi ke model Pydantic
    status_code=status.HTTP_200_OK)
def get_scraped_result():
    try:
        logger.info("Start fetching scraped data from the database")
        results = list(mongo_collection.find())

        if not results:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="The result is empty"
            )

        logger.info(f"Fetched {len(results)} results from database.")
        
        # Konversi ObjectId dan tipe data lain yang tidak dapat di-serialize JSON
        for doc in results:
            # Konversi ObjectId ke string
            if '_id' in doc:
                doc['_id'] = str(doc['_id'])
            
            # Jika ada tipe data MongoDB lain yang perlu dikonversi, tambahkan di sini
            # Contoh untuk datetime
            for key, value in doc.items():
                if isinstance(value, datetime):
                    doc[key] = value.isoformat()
        
        # Kembalikan hasil mentah tanpa konversi ke Pydantic
        return results
    except Exception as e:
        logger.error(f"Error fetching data: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error fetching data")
@app.get("/health")
async def health_check():
    try:
        ping_result = mongo_collection.database.command('ping')
        if ping_result.get('ok') == 1:
            return {
                "status": "healthy",
                "mongodb_connection": "ok",
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {
                "status": "degraded",
                "mongodb_connection": "failed",
                "timestamp": datetime.now().isoformat()
            }
    except Exception as e:
        return {
            "status": "unhealthy",
            "mongodb_connection": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

if __name__ == "__main__":
    uvicorn.run(
        app=app,
        host=app_host,
        port=app_port
    )