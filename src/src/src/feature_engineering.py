"""Feature engineering for fraud detection."""

import pandas as pd
import numpy as np
from typing import List


class FeatureEngineer:
    """Create engineered features for fraud detection."""
    
    def __init__(self):
        self.amount_stats = {}
    
    def fit(self, df: pd.DataFrame):
        """Calculate statistics for feature engineering."""
        self.amount_stats['mean'] = df['Amount'].mean()
        self.amount_stats['std'] = df['Amount'].std()
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply feature engineering."""
        df = df.copy()
        
        # Amount-based features
        df['Amount_Log'] = np.log1p(df['Amount'])
        df['Amount_ZScore'] = (df['Amount'] - self.amount_stats['mean']) / self.amount_stats['std']
        df['Amount_Bin'] = pd.qcut(df['Amount'], q=5, labels=['very_low', 'low', 'medium', 'high', 'very_high'])
        
        # Interaction features (for V columns if present)
        v_cols = [c for c in df.columns if c.startswith('V')]
        if len(v_cols) >= 2:
            df['V_Mean'] = df[v_cols].mean(axis=1)
            df['V_Std'] = df[v_cols].std(axis=1)
            df['V_Max'] = df[v_cols].max(axis=1)
            df['V_Min'] = df[v_cols].min(axis=1)
        
        # Velocity features (if Time is present)
        if 'Time' in df.columns:
            df['Time_Hour'] = (df['Time'] / 3600) % 24
            df['Time_Day'] = (df['Time'] / (3600 * 24))
        
        return df
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform."""
        self.fit(df)
        return self.transform(df)
