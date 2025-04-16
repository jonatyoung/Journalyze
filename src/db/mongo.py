import os
from pymongo import MongoClient
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def connect_to_mongodb():
    """
    Establish connection to MongoDB with authentication support
    """

    host = os.getenv('MONGO_HOST') 
    port = os.getenv('MONGO_PORT') 
    database_name=str(os.getenv('MONGO_DB'))
    collection_name=os.getenv('MONGO_COLLECTION')
    username=os.getenv('MONGO_USERNAME')
    password=os.getenv('MONGO_PASSWORD')

    uri = f"mongodb://{host}:{port}"

    try:
        connection_options = {
            'socketTimeoutMS': 3600000,
            'connectTimeoutMS': 30000,
            'serverSelectionTimeoutMS': 30000
        }
        
        if username and password:
            client = MongoClient(
                uri,
                username=username,
                password=password,
                authSource=database_name,
                **connection_options
            )
        else:
            client = MongoClient(uri, **connection_options)
        
        client.admin.command('ping')
        
        db = client[database_name]
        collection = db[collection_name]
        
        logger.info("✅ Successfully connected to MongoDB!")
        return collection
    
    except Exception as e:
        logger.error(f"MongoDB Connection Error: {e}")
        
        if 'authentication' in str(e).lower():
            logger.error("🔒 Authentication Issues:")
            logger.error("1. Verify username and password")
            logger.error("2. Check database access permissions")
            logger.error("3. Verify authentication source database")
        
        return None