import mlflow
from train import BERTopicTrainer
from dotenv import load_dotenv
import os

load_dotenv()

host = os.getenv("MLFLOW_HOST")
port = os.getenv("MLFLOW_PORT")
mlflow_uri = f"http://{host}:{port}"

db_host = os.getenv("DB_SERVICE_HOST")
db_port = os.getenv("DB_SERVICE_PORT")
db_endpoint = f"http://{db_host}:{db_port}/scraped-results"

mlflow.set_tracking_uri(uri=mlflow_uri)
mlflow.set_experiment("Journalyze experiment")

trainer = BERTopicTrainer(
    n_neighbors=15,
    n_components=10,
    min_cluster_size=25
)

trainer.load_data(api_endpoint=db_endpoint)

with mlflow.start_run():
    mlflow.log_param("n_neighbors", trainer.n_neighbors)
    mlflow.log_param("n_components", trainer.n_components)
    mlflow.log_param("min_cluster_size", trainer.min_cluster_size)

    # Train
    trainer.train()

    trainer.calculate_coherence_score()

    # Get evaluation metrics
    metrics = trainer.get_evaluation_metrics()
    print("Evaluation metrics:")
    
    for metric, value in metrics.items():
        print(f"- {metric}: {value}")
        
        # Skip topic_distribution (since it's a dict) and log only numerical metrics
        if isinstance(value, dict):
            print(f"Skipping {metric} since it is a dictionary.")
        elif value is not None:
            try:
                mlflow.log_metric(metric, float(value))
            except ValueError:
                print(f"Skipping non-numeric metric {metric}")
                
    # Save model
    model_path = trainer.save_model()
    print(f"Model saved to: {model_path}")
    mlflow.log_artifacts(model_path, artifact_path="model")

print("Training is completed and tracked with MLflow.")