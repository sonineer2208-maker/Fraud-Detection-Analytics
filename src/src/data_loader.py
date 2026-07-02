"""Data loading and preprocessing utilities."""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """Handle data loading, splitting, and preprocessing."""
    
    def __init__(self, config: dict):
        self.config = config
        self.scaler = StandardScaler()
        self.target_col = config['features']['target_column']
        self.drop_cols = config['features']['drop_columns']
        
    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load raw data from CSV."""
        logger.info(f"Loading data from {filepath}")
        df = pd.read_csv(filepath)
        logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
        return df
    
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Basic preprocessing."""
        df = df.copy()
        
        # Drop specified columns
        if self.drop_cols:
            df = df.drop(columns=self.drop_cols, errors='ignore')
            logger.info(f"Dropped columns: {self.drop_cols}")
        
        # Handle missing values
        missing = df.isnull().sum()
        if missing.any():
            logger.warning(f"Missing values found:\n{missing[missing > 0]}")
            df = df.fillna(df.median())
        
        return df
    
    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split into train/test sets."""
        X = df.drop(columns=[self.target_col])
        y = df[self.target_col]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config['data']['test_size'],
            random_state=self.config['data']['random_state'],
            stratify=y
        )
        
        logger.info(f"Train set: {len(X_train)} samples")
        logger.info(f"Test set: {len(X_test)} samples")
        logger.info(f"Fraud ratio in train: {y_train.mean():.4f}")
        
        return X_train, X_test, y_train, y_test
    
    def scale_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Scale features using StandardScaler."""
        if self.config['features']['scale_features']:
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            logger.info("Features scaled using StandardScaler")
            return X_train_scaled, X_test_scaled
        return X_train.values, X_test.values
    
    def get_feature_names(self, X: pd.DataFrame) -> list:
        """Return feature names."""
        return list(X.columns)
