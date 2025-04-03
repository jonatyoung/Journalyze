from scholarly import scholarly
import pandas as pd
import numpy as np
import json
from datetime import datetime
import logging
import os
from dotenv import load_dotenv
import re
import string
import torch
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from transformers import BertTokenizer, BertModel
from tqdm.auto import tqdm

class Pipeline:
    def __init__(self, query, max_results=10):
        self.query = query
        self.max_results = max_results
        self.model_name = "bert-base-uncased"
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.model = BertModel.from_pretrained(self.model_name)
        self.model.eval()
        self.logger = self.setup_logging()

    def setup_logging(self):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        return logging.getLogger(__name__)

    def preprocess_text(self, text):
        """
        Preprocess text by lowercasing, removing punctuation, 
        removing numbers, tokenizing, removing stopwords, and lemmatizing
        """

        text = text.lower()
        text = re.sub(f"[{string.punctuation}]", "", text)  # Remove punctuation
        text = re.sub(r'\d+', '', text)  # Remove numbers
        tokens = word_tokenize(text)  # Tokenization
        tokens = [word for word in tokens if word not in stopwords.words('english')]  # Remove stopwords
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]  # Lemmatization
        return ' '.join(tokens)

    def get_bert_embedding(self, text):
        """
        Generate BERT embedding for a given text
        """
        tokens = self.tokenizer(text, padding='max_length', truncation=True, max_length=20, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model(**tokens)
        
        cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
        return cls_embedding.tolist()  # Convert to list for JSON serialization

    def scrape_google_scholar(self):
        """
        Scrape publication data from Google Scholar with preprocessing and embedding
        """
        publications = []
        
        try:
            search_query = scholarly.search_pubs(self.query)
            
            for _ in range(self.max_results):
                try:
                    pub = next(search_query)
                    
                    # Extract publication info
                    title = pub.get('bib', {}).get('title', 'N/A')
                    abstract = pub.get('bib', {}).get('abstract', 'N/A')
                    
                    # Preprocess title and abstract
                    cleaned_title = self.preprocess_text(title)
                    cleaned_abstract = self.preprocess_text(abstract) if abstract != 'N/A' else 'N/A'
                    
                    # Generate embeddings
                    title_embedding = self.get_bert_embedding(cleaned_title)
                    abstract_embedding = self.get_bert_embedding(cleaned_abstract) if cleaned_abstract != 'N/A' else None
                    
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
                    self.logger.warning("Publication search completed before reaching max_results")
                    break
        
        except Exception as e:
            self.logger.error(f"Scraping Error: {e}")
        
        return publications

    def save_to_json(self, results, output_file='preprocessed_scholar_results.json'):
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4, default=str)
        self.logger.info(f"📄 Results saved to {output_file}")

    def run_pipeline(self):
        try:
            results = self.scrape_google_scholar()
            
            if results:
                self.logger.info(f"Successfully scraped {len(results)} documents")
                self.save_to_json(results)

                return results
            else:
                self.logger.warning("No results to save")
                return []
        
        except Exception as e:
            self.logger.error(f"Pipeline Error: {e}")
            return []