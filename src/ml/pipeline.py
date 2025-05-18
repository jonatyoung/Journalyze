from scholarly import scholarly  
import json  
from datetime import datetime  
import logging  
import re  
import string  
import torch  
from nltk.tokenize import word_tokenize  
from nltk.corpus import stopwords  
from nltk.stem import WordNetLemmatizer  
from transformers import BertTokenizer, BertModel  
import time  
import random as rnd  
from scrap import Scrap

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
    def setup_logging(self):  
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')  
        return logging.getLogger(__name__)  

    def preprocess_text(self, text):  
        """  
        Preprocess text by lowercasing, removing punctuation,   
        removing numbers, tokenizing, removing stopwords, and lemmatizing.  
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
        Generate BERT embedding for a given text.  
        """  
        tokens = self.tokenizer(text, padding='max_length', truncation=True, max_length=20, return_tensors="pt")  
        
        with torch.no_grad():  
            outputs = self.model(**tokens)  
        
        cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()  
        return cls_embedding.tolist()  # Convert to list for JSON serialization  

    def preprocess_publications(self, publications):  
        """  
        Preprocess the scraped publications and generate embeddings.  
        """  
        processed_publications = []  
        
        for pub in publications:  
            title = pub['title']  
            abstract = pub['abstract']  
            
            cleaned_title = self.preprocess_text(title)  
            cleaned_abstract = self.preprocess_text(abstract) if abstract != 'N/A' else 'N/A'  

            # embedding
            title_embedding = self.get_bert_embedding(cleaned_title)  
            abstract_embedding = self.get_bert_embedding(cleaned_abstract) if cleaned_abstract != 'N/A' else None  

            publication_info = {  
                'original_title': title,  
                'original_abstract': abstract,  
                'cleaned_title': cleaned_title,  
                'cleaned_abstract': cleaned_abstract,  
                'title_embedding': title_embedding,  
                'abstract_embedding': abstract_embedding,  
                'venue': pub['journal_conference_name'], 
                'authors': pub['authors'],  
                'year': pub['year'],  
                'doi': pub['doi'],  
                'scrape_timestamp': datetime.now().isoformat()  
            }  

            processed_publications.append(publication_info)  
            time.sleep(rnd.randint(1, 10))  

        return processed_publications  

    def save_to_json(self, results, output_file):  
        """  
        Save results to a JSON file.  
        """  
        with open(output_file, 'w', encoding='utf-8') as f:  
            json.dump(results, f, ensure_ascii=False, indent=4, default=str)  
        self.logger.info(f"📄 Results saved to {output_file}")  

    def run_pipeline(self):  
        try:  
            # scrape data  
            scraper = Scrap(self.query, self.max_results)  
            scrap_results = scraper.scrape_google_scholar()  
            
            self.logger.info(f"Successfully scraped {len(scrap_results)} documents")  
            self.save_to_json(scrap_results, 'scraped_scholar_results.json')  

            # preprocessing
            preprocessed_results = self.preprocess_publications(scrap_results)  
            self.save_to_json(preprocessed_results, 'preprocessed_scholar_results.json')  

            return scrap_results, preprocessed_results  

        except Exception as e:  
            self.logger.error(f"Pipeline Error: {e}")  
            return []  