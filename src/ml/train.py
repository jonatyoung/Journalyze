import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path
import logging
from typing import List, Dict, Tuple, Optional, Union, Any
import requests
import umap.umap_ as umap
from bertopic import BERTopic
from gensim.models.coherencemodel import CoherenceModel
import gensim.corpora as corpora
from gensim.models import TfidfModel

class BERTopicTrainer:
    
    def __init__(
        self,
        base_dir: str = "./",
        n_neighbors: int = 15,
        n_components: int = 10,
        min_cluster_size: int = 25,
        random_state: int = 42
    ):
        self.base_dir = base_dir
        self.data_dir = os.path.join(base_dir, "data/")
        self.model_dir = os.path.join(base_dir, "models/")
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.min_cluster_size = min_cluster_size
        self.random_state = random_state
        
        # Create directories if they don't exist
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('BERTopicTrainer')
        
        # Initialize model and results storage
        self.topic_model = None
        self.topics = None
        self.probs = None
        self.embeddings = None
        self.texts = None
        self.coherence_score = None
        self.topic_info = None
        
    def load_data(self, 
          api_endpoint: Optional[str] = None,
          api_results: Optional[List[Dict[str, Any]]] = None,
          file_path: Optional[str] = None, 
          text_column: str = 'cleaned_title',
          embedding_column: str = 'title_embedding',
          embedding_path: Optional[str] = None) -> None:
        
        if api_endpoint is not None:
            self.logger.info(f"Loading data from API endpoint: {api_endpoint}")
            response = requests.get(api_endpoint)
            if response.status_code != 200:
                self.logger.error(f"Failed to fetch data: Status code {response.status_code}")
                raise ValueError(f"API request failed with status code {response.status_code}")
            
            api_results = response.json()
            
        if api_results is not None:
            self.logger.info("Processing API results")
            
            # Extract texts from API results
            self.texts = [doc.get(text_column, "") for doc in api_results if text_column in doc]
            
            # Extract embeddings from API results
            embeddings_list = [doc.get(embedding_column, []) for doc in api_results if embedding_column in doc]
            if embeddings_list:
                self.embeddings = np.array(embeddings_list)
            else:
                self.logger.warning(f"No embeddings found with key '{embedding_column}'")
                raise ValueError(f"No embeddings found with key '{embedding_column}'")
                
        elif file_path is not None:
            self.logger.info(f"Loading data from local file: {file_path}")
            file_extension = os.path.splitext(file_path)[1].lower()
            
            if file_extension == '.csv':
                df = pd.read_csv(file_path)
            elif file_extension == '.parquet':
                df = pd.read_parquet(file_path)
            elif file_extension in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path)
            else:
                self.logger.error(f"Unsupported file format: {file_extension}")
                raise ValueError(f"Unsupported file format: {file_extension}")
                
            if text_column not in df.columns:
                self.logger.error(f"Text column '{text_column}' not found in dataframe")
                raise ValueError(f"Text column '{text_column}' not found in dataframe")
                
            self.texts = df[text_column].tolist()
            
            if embedding_path is not None:
                self.logger.info(f"Loading embeddings from: {embedding_path}")
                self.embeddings = np.load(embedding_path)
            elif embedding_column in df.columns:
                self.logger.info(f"Using embeddings from dataframe column: {embedding_column}")
                # Extract embeddings from the dataframe if they're stored as lists
                try:
                    self.embeddings = np.array(df[embedding_column].tolist())
                except:
                    self.logger.error(f"Failed to convert '{embedding_column}' column to numpy array")
                    raise ValueError(f"Failed to convert '{embedding_column}' column to numpy array")
            else:
                self.logger.error("No embeddings source provided")
                raise ValueError("Either embedding_path or embedding_column must be provided")
        else:
            self.logger.error("No data source provided")
            raise ValueError("Either api_endpoint, api_results, or file_path must be provided")
        
        # Ensure we have the same number of texts and embeddings
        if len(self.texts) != len(self.embeddings):
            self.logger.warning(f"Mismatch in document count: {len(self.texts)} texts vs {len(self.embeddings)} embeddings")
            # Keep only the matching number of documents
            min_count = min(len(self.texts), len(self.embeddings))
            self.texts = self.texts[:min_count]
            self.embeddings = self.embeddings[:min_count]
            
        # Verify we have data to work with
        if len(self.texts) == 0:
            self.logger.error("No text data loaded")
            raise ValueError("No text data loaded")
            
        self.logger.info(f"Loaded {len(self.texts)} documents and embeddings with shape {self.embeddings.shape}")
        
    def _create_umap_model(self) -> umap.UMAP:
        
        return umap.UMAP(
            n_neighbors=self.n_neighbors,
            n_components=self.n_components,
            metric='cosine',
            random_state=self.random_state
        )
    
    def _tokenized_texts(self) -> List[List[str]]:
        
        # Simple whitespace tokenization - can be improved with better tokenization
        return [text.lower().split() for text in self.texts]
    
    def train(self) -> None:

        if self.texts is None or self.embeddings is None:
            raise ValueError("Data not loaded. Call load_data() first.")
            
        self.logger.info("Creating UMAP model for dimensionality reduction")
        umap_model = self._create_umap_model()
        
        self.logger.info("Training BERTopic model")
        self.topic_model = BERTopic(
            umap_model=umap_model, 
            embedding_model=None,  # Using precomputed embeddings
            min_topic_size=self.min_cluster_size,
            verbose=True
        )
        
        self.topics, self.probs = self.topic_model.fit_transform(
            self.texts, 
            embeddings=self.embeddings
        )
        
        self.topic_info = self.topic_model.get_topic_info()
        self.logger.info(f"Model training complete. Found {len(self.topic_info) - 1} topics")
        
    def calculate_coherence_score(self, coherence_metric: str = 'c_v') -> float:
        if self.topic_model is None:
            raise ValueError("Model not trained. Call train() first.")
            
        self.logger.info(f"Calculating coherence score using {coherence_metric} metric")
        
        # Get topic words (excluding outlier topic -1)
        topics_dict = {topic_id: [word for word, _ in self.topic_model.get_topic(topic_id)]
                    for topic_id in set(self.topics) if topic_id != -1}
        
        if len(topics_dict) == 0:
            self.logger.warning("No meaningful topics found for coherence calculation")
            return 0.0
            
        # Convert topics to list format that CoherenceModel expects
        topic_words = [topics_dict[topic_id] for topic_id in sorted(topics_dict.keys())]
        
        # Prepare tokenized texts for coherence model
        tokenized_texts = self._tokenized_texts()
        
        # Create dictionary and corpus for coherence calculation
        dictionary = corpora.Dictionary(tokenized_texts)
        corpus = [dictionary.doc2bow(text) for text in tokenized_texts]
        
        # Create coherence model
        try:
            coherence_model = CoherenceModel(
                topics=topic_words, 
                texts=tokenized_texts, 
                dictionary=dictionary,
                corpus=corpus,
                coherence=coherence_metric
            )
            
            self.coherence_score = coherence_model.get_coherence()
            self.logger.info(f"Coherence score ({coherence_metric}): {self.coherence_score}")
            
        except Exception as e:
            self.logger.error(f"Error calculating coherence score: {str(e)}")
            self.coherence_score = None
        
        return self.coherence_score
    
    def save_model(self, model_name: str = "bertopic_model", db_service = None) -> str:

        if self.topic_model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        model_metadata = {
            "coherence_score": self.coherence_score,
            "num_topics": len(self.topic_info) - 1,  # Exclude outlier topic
            "topic_sizes": self.topic_info['Count'].tolist(),
            "hyperparameters": {
                "n_neighbors": self.n_neighbors,
                "n_components": self.n_components,
                "min_cluster_size": self.min_cluster_size
            }
        }
        
        if db_service is not None:
            self.logger.info("Saving model via database service")
            try:
                # Serialize model
                model_binary = pickle.dumps(self.topic_model)
                # Save to database via service
                db_service.save_model(model_binary, model_name, model_metadata)
                return f"Model '{model_name}' saved to database"
            except Exception as e:
                self.logger.error(f"Error saving model to database: {str(e)}")
                raise
        else:
            # Save locally
            model_path = os.path.join(self.model_dir, f"{model_name}.pkl")
            metadata_path = os.path.join(self.model_dir, f"{model_name}_metadata.pkl")
            
            try:
                with open(model_path, 'wb') as f:
                    pickle.dump(self.topic_model, f)
                    
                with open(metadata_path, 'wb') as f:
                    pickle.dump(model_metadata, f)
                    
                self.logger.info(f"Model saved to {model_path}")
                self.logger.info(f"Model metadata saved to {metadata_path}")
                return model_path
            except Exception as e:
                self.logger.error(f"Error saving model to file: {str(e)}")
                raise
    
    def load_model(self, model_name: str = "bertopic_model", db_service = None) -> BERTopic:
        
        if db_service is not None:
            self.logger.info(f"Loading model '{model_name}' from database service")
            try:
                model_binary = db_service.get_model(model_name)
                if model_binary is None:
                    raise ValueError(f"Model '{model_name}' not found in database")
                    
                self.topic_model = pickle.loads(model_binary)
            except Exception as e:
                self.logger.error(f"Error loading model from database: {str(e)}")
                raise
        else:
            model_path = os.path.join(self.model_dir, f"{model_name}.pkl")
            self.logger.info(f"Loading model from {model_path}")
            
            if not os.path.exists(model_path):
                error_msg = f"Model file not found: {model_path}"
                self.logger.error(error_msg)
                raise FileNotFoundError(error_msg)
                
            try:
                with open(model_path, 'rb') as f:
                    self.topic_model = pickle.load(f)
            except Exception as e:
                self.logger.error(f"Error loading model from file: {str(e)}")
                raise
                
        return self.topic_model
    
        """
        Generate topic visualization
        
        Args:
            viz_type: Type of visualization ('barchart', 'topics', 'hierarchy', etc.)
            top_n: Number of top topics to include
            
        Returns:
            The visualization object
        """
        if self.topic_model is None:
            raise ValueError("Model not trained. Call train() first.")
            
        try:
            if viz_type == 'barchart':
                return self.topic_model.visualize_barchart(top_n_topics=top_n)
            elif viz_type == 'topics':
                return self.topic_model.visualize_topics()
            elif viz_type == 'hierarchy':
                return self.topic_model.visualize_hierarchy()
            elif viz_type == 'heatmap':
                return self.topic_model.visualize_heatmap()
            elif viz_type == 'documents':
                return self.topic_model.visualize_documents(self.embeddings)
            else:
                raise ValueError(f"Visualization type '{viz_type}' not supported")
        except Exception as e:
            self.logger.error(f"Error generating visualization: {str(e)}")
            raise
    
    def get_evaluation_metrics(self) -> Dict[str, Any]:
        
        if self.topic_model is None:
            raise ValueError("Model not trained. Call train() first.")
            
        try:
            outlier_count = 0
            if -1 in self.topic_info['Topic'].values:
                outlier_count = self.topic_info.loc[self.topic_info['Topic'] == -1, 'Count'].values[0]
                
            metrics = {
                "coherence_score": self.coherence_score,
                "num_topics": len(self.topic_info) - 1,  # Exclude outlier topic
                "outlier_percentage": outlier_count / len(self.topics) if len(self.topics) > 0 else 0,
                "topic_distribution": dict(zip(self.topic_info['Topic'].tolist(), self.topic_info['Count'].tolist())),
                "document_count": len(self.texts) if self.texts is not None else 0,
                "avg_topic_size": self.topic_info[self.topic_info['Topic'] != -1]['Count'].mean() if len(self.topic_info) > 1 else 0
            }
            
            return metrics
        except Exception as e:
            self.logger.error(f"Error calculating evaluation metrics: {str(e)}")
            raise


# Example usage
if __name__ == "__main__":
    # Initialize trainer
    trainer = BERTopicTrainer(base_dir="./")
    
    try:
        # Load from API endpoint
        trainer.load_data(api_endpoint="http://172.19.0.3:8002/scraped-results")
        
        # Train model
        trainer.train()
        
        # Calculate coherence score
        coherence_score = trainer.calculate_coherence_score()
        print(f"Model coherence score: {coherence_score}")
        
        # Get topic information
        topic_info = trainer.topic_model.get_topic_info()
        print("Topic information:")
        print(topic_info.head(10))
        
        # Save model
        model_path = trainer.save_model()
        print(f"Model saved to: {model_path}")
        
        # Get evaluation metrics
        metrics = trainer.get_evaluation_metrics()
        print("Evaluation metrics:")
        for metric, value in metrics.items():
            print(f"- {metric}: {value}")
        
        # Save evaluation metrics to text file
        evaluation_file = os.path.join(trainer.model_dir, "evaluation_metrics.txt")
        with open(evaluation_file, 'w') as f:
            f.write(f"BERTopic Model Evaluation Metrics\n")
            f.write(f"==============================\n")
            f.write(f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Write general metrics
            f.write(f"General Metrics:\n")
            f.write(f"- Coherence Score: {metrics['coherence_score']}\n")
            f.write(f"- Number of Topics: {metrics['num_topics']}\n")
            f.write(f"- Document Count: {metrics['document_count']}\n")
            f.write(f"- Average Topic Size: {metrics['avg_topic_size']:.2f}\n")
            f.write(f"- Outlier Percentage: {metrics['outlier_percentage']:.2%}\n\n")
            
            # Write topic distribution
            f.write(f"Topic Distribution:\n")
            for topic_id, count in metrics['topic_distribution'].items():
                if topic_id == -1:
                    f.write(f"- Outlier Topic (-1): {count} documents\n")
                else:
                    # Get the top words for this topic if available
                    topic_words = ""
                    try:
                        top_words = [word for word, _ in trainer.topic_model.get_topic(topic_id)[:5]]
                        topic_words = ", ".join(top_words)
                    except:
                        topic_words = "N/A"
                    
                    f.write(f"- Topic {topic_id}: {count} documents - Top words: {topic_words}\n")
            
            # Add detailed topic information
            f.write(f"\nDetailed Topic Information:\n")
            f.write(f"-------------------------\n")
            # Process each topic (excluding outliers)
            for _, row in topic_info[topic_info['Topic'] != -1].iterrows():
                topic_id = row['Topic']
                name = row['Name']
                count = row['Count']
                representation = row['Representation']
                f.write(f"Topic {topic_id}:\n")
                f.write(f"  Name: {name}\n")
                f.write(f"  Document Count: {count}\n")
                f.write(f"  Representation: {representation}\n")
                
                # Add top 10 terms with their weights
                f.write(f"  Top Terms (with weights):\n")
                try:
                    top_terms = trainer.topic_model.get_topic(topic_id)[:10]
                    for term, weight in top_terms:
                        f.write(f"    - {term}: {weight:.4f}\n")
                except Exception as e:
                    f.write(f"    Error retrieving terms: {str(e)}\n")
                
                # Add representative documents if available
                try:
                    f.write(f"  Representative Documents (sample):\n")
                    docs_per_topic = trainer.topic_model.get_representative_docs(topic_id)
                    # Limit to 3 sample documents
                    for i, doc in enumerate(docs_per_topic[:3]):
                        f.write(f"    {i+1}. {doc[:100]}{'...' if len(doc) > 100 else ''}\n")
                except Exception as e:
                    f.write(f"    Error retrieving documents: {str(e)}\n")
                
                f.write("\n")
            
            # Write hyperparameters
            f.write(f"\nModel Hyperparameters:\n")
            f.write(f"- n_neighbors: {trainer.n_neighbors}\n")
            f.write(f"- n_components: {trainer.n_components}\n")
            f.write(f"- min_cluster_size: {trainer.min_cluster_size}\n")
            f.write(f"- random_state: {trainer.random_state}\n")
        
        print(f"Evaluation metrics saved to: {evaluation_file}")
        
    except Exception as e:
        print(f"Error in topic modeling process: {str(e)}")
        # Save error to file
        error_file = os.path.join(trainer.model_dir, "error_log.txt")
        with open(error_file, 'w') as f:
            f.write(f"Error occurred at {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Error message: {str(e)}\n")
            import traceback
            f.write(traceback.format_exc())
        print(f"Error details saved to: {error_file}")