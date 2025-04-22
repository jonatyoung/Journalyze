import time  
import random as rnd  
from scholarly import scholarly  
import logging  

class Scrap:  
    def __init__(self, query, max_results=10):  
        self.query = query  
        self.max_results = max_results  
        self.logger = logging.getLogger(__name__)  

    def scrape_google_scholar(self):  
        """  
        Scrape publication data from Google Scholar.  
        """  
        publications = []  
        
        try:  
            search_query = scholarly.search_pubs(self.query)  
            
            for _ in range(self.max_results):  
                try:  
                    pub = next(search_query)  

                    # Extract publication info  
                    publication_info = {  
                        "title": pub.get('bib', {}).get('title', 'N/A'),  
                        "abstract": pub.get('bib', {}).get('abstract', 'N/A'),  
                        "authors": pub.get('bib', {}).get('author', ['N/A']),  
                        "journal_conference_name": pub.get('bib', {}).get('venue', 'N/A'),   
                        "publisher": pub.get('source', 'N/A'),  
                        "year": pub.get('bib', {}).get('pub_year', 'N/A'),  
                        "doi": pub.get('pub_url', 'N/A'),  
                        "group_name": "Journalyze"  
                    }  
                    
                    publications.append(publication_info)  
                    time.sleep(rnd.randint(1, 10))  # Sleep to avoid overloading the server  
                
                except StopIteration:  
                    self.logger.warning("Publication search completed before reaching max_results")  
                    break  
        
        except Exception as e:  
            self.logger.error(f"Scraping Error: {e}")  
        
        return publications  