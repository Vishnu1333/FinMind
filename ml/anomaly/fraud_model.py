from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, precision_recall_fscore_support
import json
import os
import joblib
from datetime import datetime
import hashlib

class FraudDetectionModel:
    def __init__(self, contamination=0.01):
        self.model = IsolationForest(contamination=contamination, random_state=42)
        
    def train(self, X_train):
        self.model.fit(X_train)
        
    def predict(self, X):
        # Isolation forest returns 1 for inliers, -1 for outliers
        preds = self.model.predict(X)
        return (preds == -1).astype(int)
        
    def evaluate(self, X_test, y_test):
        preds = self.predict(X_test)
        metrics = precision_recall_fscore_support(y_test, preds, average='binary')
        return {
            "precision": metrics[0],
            "recall": metrics[1],
            "f1": metrics[2]
        }
        
    def save(self, path, model_name, dataset_name, metrics):
        os.makedirs(path, exist_ok=True)
        model_file = os.path.join(path, f"{model_name}.pkl")
        joblib.dump(self.model, model_file)
        
        # Calculate checksum
        with open(model_file, "rb") as f:
            checksum = hashlib.md5(f.read()).hexdigest()
            
        registry_entry = {
            "model_name": model_name,
            "version": "1.0",
            "dataset": dataset_name,
            "training_timestamp": datetime.utcnow().isoformat(),
            "metrics": metrics,
            "artifact_path": model_file,
            "artifact_checksum": checksum,
            "status": "verified"
        }
        
        registry_file = os.path.join(path, f"{model_name}_registry.json")
        with open(registry_file, "w") as f:
            json.dump(registry_entry, f, indent=4)
        
        return registry_entry
