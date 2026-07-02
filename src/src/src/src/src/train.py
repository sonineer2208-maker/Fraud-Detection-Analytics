"""Training pipeline."""

import os
import yaml
import joblib
import logging
from typing import Dict, Any
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from .models import ModelFactory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Trainer:
    """Handle model training and cross-validation."""
    
    def __init__(self, config: dict):
        self.config = config
        self.models_dir = config['paths']['models_dir']
        os.makedirs(self.models_dir, exist_ok=True)
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              model_name: str, use_smote: bool = True) -> Any:
        """Train a single model."""
        logger.info(f"Training {model_name}...")
        
        model = ModelFactory.get_model(model_name, self.config)
        
        if use_smote:
            # Use SMOTE for handling imbalance
            pipeline = ImbPipeline([
                ('smote', SMOTE(random_state=42)),
                ('classifier', model)
            ])
        else:
            pipeline = model
        
        # Cross-validation
        cv = StratifiedKFold(
            n_splits=self.config['training']['cv_folds'],
            shuffle=True,
            random_state=42
        )
        
        scores = cross_val_score(
            pipeline, X_train, y_train,
            cv=cv,
            scoring=self.config['training']['scoring'],
            n_jobs=-1
        )
        
        logger.info(f"CV ROC-AUC: {scores.mean():.4f} (+/- {scores.std()*2:.4f})")
        
        # Fit on full training data
        pipeline.fit(X_train, y_train)
        
        return pipeline
    
    def save_model(self, model: Any, model_name: str):
        """Save trained model."""
        filepath = os.path.join(self.models_dir, f"{model_name}.joblib")
        joblib.dump(model, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def train_all(self, X_train: np.ndarray, y_train: np.ndarray,
                  model_names: list, use_smote: bool = True) -> Dict[str, Any]:
        """Train multiple models."""
        trained_models = {}
        
        for name in model_names:
            try:
                model = self.train(X_train, y_train, name, use_smote)
                self.save_model(model, name)
                trained_models[name] = model
            except Exception as e:
                logger.error(f"Failed to train {name}: {e}")
        
        return trained_models
