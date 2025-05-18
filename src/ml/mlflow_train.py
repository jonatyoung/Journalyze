import mlflow
from train import BERTopicTrainer
from dotenv import load_dotenv
import os
import shutil
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from prometheus_client import Summary

TRAINING_DURATION = Summary('training_duration_seconds', 'Time spent training BERTopic')

class MLflowTrainer:
    def __init__(self):
        load_dotenv()
        
        self.logger = logging.getLogger(__name__)
        
        host = os.getenv("MLFLOW_HOST")
        port = os.getenv("MLFLOW_PORT")
        self.mlflow_uri = f"http://{host}:{port}"
        
        db_host = os.getenv("DB_SERVICE_HOST")
        db_port = os.getenv("DB_SERVICE_PORT")
        self.db_endpoint = f"http://{db_host}:{db_port}/scraped-results"
        
        mlflow.set_tracking_uri(uri=self.mlflow_uri)
        mlflow.set_experiment("Journalyze experiment")
        
        self.default_params = {
            "n_neighbors": 15,
            "n_components": 10,
            "min_cluster_size": 25
        }
        
        self.last_training_metrics = None
    
    @TRAINING_DURATION.time()
    def train_model(self, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if params is None:
            params = self.default_params
            
        try:
            trainer = BERTopicTrainer(
                n_neighbors=params.get("n_neighbors", self.default_params["n_neighbors"]),
                n_components=params.get("n_components", self.default_params["n_components"]),
                min_cluster_size=params.get("min_cluster_size", self.default_params["min_cluster_size"])
            )
            
            trainer.load_data(api_endpoint=self.db_endpoint)
            
            with mlflow.start_run():
                mlflow.log_param("n_neighbors", trainer.n_neighbors)
                mlflow.log_param("n_components", trainer.n_components)
                mlflow.log_param("min_cluster_size", trainer.min_cluster_size)
                
                trainer.train()
                
                trainer.calculate_coherence_score()
                
                metrics = trainer.get_evaluation_metrics()
                self.logger.info("Evaluation metrics:")
                
                for metric, value in metrics.items():
                    if isinstance(value, dict):
                        self.logger.info(f"Skipping {metric} since it is a dictionary.")
                    elif value is not None:
                        try:
                            mlflow.log_metric(metric, float(value))
                            self.logger.info(f"- {metric}: {value}")
                        except ValueError:
                            self.logger.info(f"Skipping non-numeric metric {metric}")
                
                model_path = trainer.save_model()
                self.logger.info(f"Model saved to: {model_path}")
                mlflow.log_artifacts(model_path, artifact_path="model")
                
                if os.path.exists(model_path) and os.path.isdir(model_path):
                    shutil.rmtree(model_path)
            
            self.last_training_metrics = metrics
            self.logger.info("Training is completed and tracked with MLflow.")
            return metrics
        
        except Exception as e:
            self.logger.error(f"Error during model training: {str(e)}")
            raise
    
    def get_latest_metrics(self) -> Dict[str, Any]:
        try:
            client = mlflow.tracking.MlflowClient()
            experiment = client.get_experiment_by_name("Journalyze experiment")
            
            if not experiment:
                return {"message": "No experiment found"}
            
            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["attribute.start_time DESC"],
                max_results=1
            )
            
            if not runs:
                return {"message": "No runs found for the experiment"}
            
            latest_run = runs[0]
            
            metrics = latest_run.data.metrics
            params = latest_run.data.params
            
            return {
                "run_id": latest_run.info.run_id,
                "status": latest_run.info.status,
                "start_time": datetime.fromtimestamp(latest_run.info.start_time/1000).isoformat(),
                "end_time": datetime.fromtimestamp(latest_run.info.end_time/1000).isoformat() if latest_run.info.end_time else None,
                "metrics": metrics,
                "parameters": params
            }
        except Exception as e:
            self.logger.error(f"Error retrieving model metrics: {str(e)}")
            raise

    def check_mlflow_connection(self) -> bool:
        try:
            mlflow.tracking.MlflowClient()
            return True
        except Exception:
            return False