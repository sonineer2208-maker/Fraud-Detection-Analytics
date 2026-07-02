"""Model definitions and factory."""

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from typing import Dict, Any


class ModelFactory:
    """Factory for creating model instances."""
    
    @staticmethod
    def get_model(model_name: str, config: dict) -> Any:
        """Get model instance by name."""
        model_configs = {
            'random_forest': ModelFactory._get_random_forest,
            'logistic_regression': ModelFactory._get_logistic_regression,
            'xgboost': ModelFactory._get_xgboost,
            'lightgbm': ModelFactory._get_lightgbm,
            'svm': ModelFactory._get_svm
        }
        
        if model_name not in model_configs:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(model_configs.keys())}")
        
        return model_configs[model_name](config)
    
    @staticmethod
    def _get_random_forest(config: dict) -> RandomForestClassifier:
        cfg = config['models']['random_forest']
        return RandomForestClassifier(
            n_estimators=cfg['n_estimators'],
            max_depth=cfg['max_depth'],
            class_weight=cfg['class_weight'],
            random_state=cfg['random_state'],
            n_jobs=-1
        )
    
    @staticmethod
    def _get_logistic_regression(config: dict) -> LogisticRegression:
        cfg = config['models']['logistic_regression']
        return LogisticRegression(
            max_iter=cfg['max_iter'],
            class_weight=cfg['class_weight'],
            random_state=cfg['random_state'],
            n_jobs=-1
        )
    
    @staticmethod
    def _get_xgboost(config: dict) -> XGBClassifier:
        cfg = config['models']['xgboost']
        return XGBClassifier(
            n_estimators=cfg['n_estimators'],
            max_depth=cfg['max_depth'],
            learning_rate=cfg['learning_rate'],
            random_state=cfg['random_state'],
            n_jobs=-1
        )
    
    @staticmethod
    def _get_lightgbm(config: dict) -> LGBMClassifier:
        return LGBMClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
    
    @staticmethod
    def _get_svm(config: dict) -> SVC:
        return SVC(
            kernel='rbf',
            class_weight='balanced',
            probability=True,
            random_state=42
        )
