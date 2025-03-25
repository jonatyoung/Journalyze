from scholarly import scholarly
import pandas as pd
import json
from pymongo import MongoClient
from datetime import datetime
import logging
import os
from dotenv import load_dotenv

load_dotenv()


# Konfigurasi logging
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
    
    Args:
        uri (str): URI koneksi MongoDB
        database_name (str): Nama database
        collection_name (str): Nama koleksi
        username (str, optional): Username untuk autentikasi
        password (str, optional): Password untuk autentikasi
    
    Returns:
        pymongo.collection.Collection: Objek koleksi MongoDB
    """
    try:
        # Opsi koneksi dengan kredensial
        connection_options = {
            'socketTimeoutMS': 30000,
            'connectTimeoutMS': 30000,
            'serverSelectionTimeoutMS': 30000
        }
        
        # Tambahkan autentikasi jika username dan password disediakan
        if username and password:
            client = MongoClient(
                uri,
                username=username,
                password=password,
                authSource=database_name,  # Database tempat user dibuat
                **connection_options
            )
        else:
            # Koneksi tanpa autentikasi
            client = MongoClient(uri, **connection_options)
        
        # Validasi koneksi dengan ping server
        client.admin.command('ping')
        
        db = client[database_name]
        collection = db[collection_name]
        
        logger.info("✅ Berhasil terhubung ke MongoDB!")
        return collection
    
    except Exception as e:
        logger.error(f"Kesalahan koneksi MongoDB: {e}")
        
        # Log detail tambahan untuk debugging
        if 'authentication' in str(e).lower():
            logger.error("🔒 Masalah Autentikasi:")
            logger.error("1. Pastikan username dan password benar")
            logger.error("2. Periksa apakah user memiliki akses ke database")
            logger.error("3. Verifikasi database sumber autentikasi (authSource)")
        
        return None

def scrape_google_scholar(query, max_results=10):
    """
    Scrape publication data from Google Scholar
    
    Args:
        query (str): Kueri pencarian
        max_results (int): Jumlah maksimal publikasi yang akan diambil
    
    Returns:
        list: Daftar publikasi yang ditemukan
    """
    publications = []
    
    try:
        search_query = scholarly.search_pubs(query)
        
        for _ in range(max_results):
            try:
                pub = next(search_query)
                
                publication_info = {
                    'title': pub.get('bib', {}).get('title', 'N/A'),
                    'abstract': pub.get('bib', {}).get('abstract', 'N/A'),
                    'venue': pub.get('bib', {}).get('venue', 'N/A'),
                    'authors': pub.get('bib', {}).get('author', ['N/A']),
                    'num_citations': pub.get('num_citations', 0),
                    'gsrank': pub.get('gsrank', -1),
                    'scrape_timestamp': datetime.now().isoformat()
                }
                
                publications.append(publication_info)
            
            except StopIteration:
                logger.warning("Pencarian publikasi selesai sebelum mencapai max_results")
                break
    
    except Exception as e:
        logger.error(f"Scraping Error: {e}")
    
    return publications

def main():
    # Konfigurasi pencarian
    query = 'machine learning'
    max_results = 20
    
    # Konfigurasi MongoDB
    mongodb_config = {
        'uri': 'mongodb://172.18.0.2:27017/',
        'database_name': 'journalyze-db',
        'collection_name': 'publications',
        # Tambahkan kredensial dari environment atau file konfigurasi
        'username': "codium",
        'password': 'rahasia321'
    }
    
    try:
        # Scrape publikasi
        results = scrape_google_scholar(query, max_results)
        
        # Koneksi ke MongoDB
        mongo_collection = connect_to_mongodb(**mongodb_config)
        
        if mongo_collection is not None:
            # Simpan ke MongoDB
            if results:
                try:
                    insert_result = mongo_collection.insert_many(results)
                    inserted_count = len(insert_result.inserted_ids)
                    logger.info(f"✅ Berhasil menyimpan {inserted_count} dokumen")
                except Exception as insert_error:
                    logger.error(f"Kesalahan saat menyimpan: {insert_error}")
            
            # Simpan ke JSON sebagai cadangan
            output_file = 'google_scholar_results.json'
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=4, default=str)
            
            logger.info(f"📄 Hasil disimpan di {output_file}")
        else:
            logger.warning("❌ Gagal terhubung ke MongoDB. Menyimpan ke JSON saja.")
            with open('google_scholar_results.json', 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=4, default=str)
    
    except Exception as e:
        logger.error(f"Kesalahan dalam proses utama: {e}")

if __name__ == "__main__":
    main()