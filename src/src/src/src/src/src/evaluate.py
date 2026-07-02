"""Model evaluation and reporting."""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve,
    average_precision_score, f1_score, precision_score, recall_score
)
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Evaluator:
    """Comprehensive model evaluation."""
    
    def __init__(self, config: dict):
        self.config = config
        self.reports_dir = config['paths']['reports_dir']
        os.makedirs(self.reports_dir, exist_ok=True)
    
    def evaluate(self, model: Any, X_test: np.ndarray, y_test: np.ndarray,
                 model_name: str, feature_names: list = None) -> dict:
        """Evaluate a single model."""
        logger.info(f"Evaluating {model_name}...")
        
        # Predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Metrics
        metrics = {
            'model': model_name,
            'roc_auc': roc_auc_score(y_test, y_prob) if y_prob is not None else None,
            'avg_precision': average_precision_score(y_test, y_prob) if y_prob is not None else None,
            'f1': f1_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred)
        }
        
        # Print report
        print(f"\n{'='*50}")
        print(f"Model: {model_name}")
        print(f"{'='*50}")
        print(classification_report(y_test, y_pred, target_names=['Normal', 'Fraud']))
        
        # Save confusion matrix
        self._plot_confusion_matrix(y_test, y_pred, model_name)
        
        # Save ROC curve
        if y_prob is not None:
            self._plot_roc_curve(y_test, y_prob, model_name)
            self._plot_precision_recall_curve(y_test, y_prob, model_name)
        
        # Feature importance
        if feature_names:
            self._plot_feature_importance(model, feature_names, model_name)
        
        return metrics
    
    def _plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str):
        """Plot confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Normal', 'Fraud'],
                   yticklabels=['Normal', 'Fraud'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        filepath = os.path.join(self.reports_dir, f'confusion_matrix_{model_name}.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved confusion matrix to {filepath}")
    
    def _plot_roc_curve(self, y_true: np.ndarray, y_prob: np.ndarray, model_name: str):
        """Plot ROC curve."""
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True)
        
        filepath = os.path.join(self.reports_dir, f'roc_curve_{model_name}.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_precision_recall_curve(self, y_true: np.ndarray, y_prob: np.ndarray, model_name: str):
        """Plot Precision-Recall curve."""
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        avg_precision = average_precision_score(y_true, y_prob)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label=f'{model_name} (AP = {avg_precision:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid(True)
        
        filepath = os.path.join(self.reports_dir, f'pr_curve_{model_name}.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_feature_importance(self, model: Any, feature_names: list, model_name: str):
        """Plot feature importance."""
        # Handle pipeline with SMOTE
        if hasattr(model, 'named_steps') and 'classifier' in model.named_steps:
            estimator = model.named_steps['classifier']
        else:
            estimator = model
        
        if hasattr(estimator, 'feature_importances_'):
            importances = estimator.feature_importances_
        elif hasattr(estimator, 'coef_'):
            importances = np.abs(estimator.coef_[0])
        else:
            return
        
        # Sort and plot top 20
        indices = np.argsort(importances)[::-1][:20]
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(indices)), importances[indices], align='center')
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Feature Importance')
        plt.title(f'Top 20 Feature Importances - {model_name}')
        plt.gca().invert_yaxis()
        
        filepath = os.path.join(self.reports_dir, f'feature_importance_{model_name}.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved feature importance to {filepath}")
    
    def compare_models(self, metrics_list: list):
        """Create comparison table and plot."""
        df = pd.DataFrame(metrics_list)
        df = df.set_index('model')
        
        print("\n" + "="*60)
        print("MODEL COMPARISON")
        print("="*60)
        print(df.round(4))
        
        # Save comparison plot
        df_plot = df[['roc_auc', 'avg_precision', 'f1', 'precision', 'recall']]
        df_plot.plot(kind='bar', figsize=(12, 6))
        plt.title('Model Comparison')
        plt.ylabel('Score')
        plt.xticks(rotation=45)
        plt.legend(loc='lower right')
        plt.tight_layout()
        
        filepath = os.path.join(self.reports_dir, 'model_comparison.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return df
