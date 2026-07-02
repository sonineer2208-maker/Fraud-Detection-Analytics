#!/usr/bin/env python3
"""
Fraud Detection Analytics - Main Pipeline

Usage:
    python main.py --data data/creditcard.csv --models random_forest xgboost
"""

import argparse
import yaml
import logging
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.data_loader import DataLoader
from src.feature_engineering import FeatureEngineer
from src.train import Trainer
from src.evaluate import Evaluator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Fraud Detection Pipeline')
    parser.add_argument('--config', default='config.yaml', help='Config file path')
    parser.add_argument('--data', required=True, help='Path to dataset CSV')
    parser.add_argument('--models', nargs='+', default=['random_forest', 'logistic_regression'],
                       help='Models to train')
    parser.add_argument('--no-smote', action='store_true', help='Disable SMOTE')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info("="*60)
    logger.info("FRAUD DETECTION ANALYTICS PIPELINE")
    logger.info("="*60)
    
    # 1. Load Data
    loader = DataLoader(config)
    df = loader.load_data(args.data)
    
    # 2. Feature Engineering
    engineer = FeatureEngineer()
    df = engineer.fit_transform(df)
    
    # One-hot encode categorical features
    df = pd.get_dummies(df, columns=['Amount_Bin'], drop_first=True)
    
    # 3. Preprocess & Split
    df = loader.preprocess(df)
    X_train, X_test, y_train, y_test = loader.split_data(df)
    
    # 4. Scale Features
    X_train_scaled, X_test_scaled = loader.scale_features(X_train, X_test)
    feature_names = loader.get_feature_names(X_train)
    
    # 5. Train Models
    trainer = Trainer(config)
    trained_models = trainer.train_all(
        X_train_scaled, y_train,
        model_names=args.models,
        use_smote=not args.no_smote
    )
    
    # 6. Evaluate Models
    evaluator = Evaluator(config)
    all_metrics = []
    
    for name, model in trained_models.items():
        metrics = evaluator.evaluate(model, X_test_scaled, y_test, name, feature_names)
        all_metrics.append(metrics)
    
    # 7. Compare Models
    if len(all_metrics) > 1:
        evaluator.compare_models(all_metrics)
    
    logger.info("\n" + "="*60)
    logger.info("PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("="*60)


if __name__ == '__main__':
    main()
