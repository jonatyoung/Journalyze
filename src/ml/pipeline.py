from scholarly import scholarly
import pandas as pd
import numpy as np
import json
from pymongo import MongoClient
from datetime import datetime
import logging
import os
from dotenv import load_dotenv

# Preprocessing and Embedding Libraries
import re
import string
import torch
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from transformers import BertTokenizer, BertModel
from tqdm.auto import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def connect_to_mongodb(
    uri='mongodb://172.18.0.2:27017/', 
    database_name='journalyze-db', 
    collection_name='publications',
    username=os.getenv('MONGODB_USERNAME'),
    password=os.getenv('MONGODB_PASSWORD')
):
    """
    Establish connection to MongoDB with authentication support
    """
    try:
        connection_options = {
            'socketTimeoutMS': 30000,
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

def preprocess_text(text):
    """
    Preprocess text by lowercasing, removing punctuation, 
    removing numbers, tokenizing, removing stopwords, and lemmatizing
    """
    text = text.lower()  # Lowercasing
    text = re.sub(f"[{string.punctuation}]", "", text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove numbers
    tokens = word_tokenize(text)  # Tokenization
    tokens = [word for word in tokens if word not in stopwords.words('english')]  # Remove stopwords
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]  # Lemmatization
    return ' '.join(tokens)

def get_bert_embedding(text, model, tokenizer):
    """
    Generate BERT embedding for a given text
    """
    tokens = tokenizer(text, padding='max_length', truncation=True, max_length=20, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**tokens)
    
    # Get [CLS] token embedding
    cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
    return cls_embedding.tolist()  # Convert to list for JSON serialization

def scrape_google_scholar(query, max_results=10):
    """
    Scrape publication data from Google Scholar with preprocessing and embedding
    """
    # Initialize BERT model and tokenizer
    model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    model.eval()

    publications = []
    
    try:
        search_query = scholarly.search_pubs(query)
        
        for _ in range(max_results):
            try:
                pub = next(search_query)
                
                # Extract publication info
                title = pub.get('bib', {}).get('title', 'N/A')
                abstract = pub.get('bib', {}).get('abstract', 'N/A')
                
                # Preprocess title and abstract
                cleaned_title = preprocess_text(title)
                cleaned_abstract = preprocess_text(abstract) if abstract != 'N/A' else 'N/A'
                
                # Generate embeddings
                title_embedding = get_bert_embedding(cleaned_title, model, tokenizer)
                abstract_embedding = get_bert_embedding(cleaned_abstract, model, tokenizer) if cleaned_abstract != 'N/A' else None
                
                publication_info = {
                    'original_title': title,
                    'original_abstract': abstract,
                    'cleaned_title': cleaned_title,
                    'cleaned_abstract': cleaned_abstract,
                    'title_embedding': title_embedding,
                    'abstract_embedding': abstract_embedding,
                    'venue': pub.get('bib', {}).get('venue', 'N/A'),
                    'authors': pub.get('bib', {}).get('author', ['N/A']),
                    'num_citations': pub.get('num_citations', 0),
                    'gsrank': pub.get('gsrank', -1),
                    'scrape_timestamp': datetime.now().isoformat()
                }
                
                publications.append(publication_info)
            
            except StopIteration:
                logger.warning("Publication search completed before reaching max_results")
                break
    
    except Exception as e:
        logger.error(f"Scraping Error: {e}")
    
    return publications

def main():
    # Search configuration
    query = 'machine learning'
    max_results = 20
    
    # MongoDB configuration
    mongodb_config = {
        'uri': 'mongodb://172.18.0.2:27017/',
        'database_name': 'journalyze-db',
        'collection_name': 'publications_preprocessed',
        'username': "codium",
        'password': 'rahasia321'
    }
    
    try:
        # Scrape publications with preprocessing and embedding
        results = scrape_google_scholar(query, max_results)
        
        # Connect to MongoDB
        mongo_collection = connect_to_mongodb(**mongodb_config)
        
        if mongo_collection is not None:
            # Save to MongoDB
            if results:
                try:
                    insert_result = mongo_collection.insert_many(results)
                    inserted_count = len(insert_result.inserted_ids)
                    logger.info(f"✅ Successfully saved {inserted_count} documents")
                except Exception as insert_error:
                    logger.error(f"Saving Error: {insert_error}")
            
            # Save to JSON as backup
            output_file = 'preprocessed_scholar_results.json'
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=4, default=str)
            
            logger.info(f"📄 Results saved to {output_file}")
        else:
            logger.warning("❌ Failed to connect to MongoDB. Saving to JSON only.")
            with open('preprocessed_scholar_results.json', 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=4, default=str)
    
    except Exception as e:
        logger.error(f"Main Process Error: {e}")

if __name__ == "__main__":
    main()