import os  
import uvicorn  
from fastapi import FastAPI  

# Import fungsi scraping dari direktori ml  
import sys  
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  
from ml.pipeline import main as scrape_main  

app = FastAPI(title="Google Scholar Scraping Service")  

@app.post("/scrape")  
async def trigger_scraping():  
    """  
    Endpoint untuk memulai proses scraping  
    """  
    try:  
        # Panggil fungsi main() dari pipeline.py  
        scrape_main()  
        return {"status": "Scraping berhasil dilakukan"}  
    except Exception as e:  
        return {"status": "Scraping gagal", "error": str(e)}  

if __name__ == "__main__":  
    uvicorn.run(app, host="0.0.0.0", port=8000)  