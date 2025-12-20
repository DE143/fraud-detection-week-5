"""
Model training and evaluation module for fraud detection.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, average_precision_score,
                           confusion_matrix, classification_report, precision_recall_curve)
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import logging
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

class FraudDetectionModel:
    """Handles model training and evaluation for fraud detection."""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.metrics = {}
        self.best_model = None
        
    def prepare_data(self, X: pd.DataFrame, y: pd.Series, test_size=0.2, handle_imbalance=True):
        """Prepare data for training with handling of class imbalance."""
        logger.info("Preparing data for training...")
        
        # Stratified train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=self.random_state
        )
        
        logger.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
        logger.info(f"Train fraud ratio: {y_train.mean():.6f}, Test fraud ratio: {y_test.mean():.6f}")
        
        if handle_imbalance:
            # Apply SMOTE only to training data
            logger.info("Applying SMOTE to handle class imbalance...")
            smote = SMOTE(random_state=self.random_state, sampling_strategy=0.1)  # 10% fraud in training
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
            
            logger.info(f"After SMOTE - Train shape: {X_train_resampled.shape}")
            logger.info(f"After SMOTE - Fraud ratio: {y_train_resampled.mean():.6f}")
            
            return X_train_resampled, X_test, y_train_resampled, y_test
        
        return X_train, X_test, y_train, y_test
    
    def train_baseline(self, X_train, y_train):
        """Train Logistic Regression as baseline model."""
        logger.info("Training Logistic Regression baseline...")
        
        model = LogisticRegression(
            random_state=self.random_state,
            max_iter=1000,
            class_weight='balanced',
            solver='liblinear'
        )
        
        model.fit(X_train, y_train)
        self.models['logistic_regression'] = model
        
        logger.info("Baseline model trained successfully")
        return model
    
    def train_random_forest(self, X_train, y_train):
        """Train Random Forest classifier."""
        logger.info("Training Random Forest...")
        
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=self.random_state,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        self.models['random_forest'] = model
        
        logger.info("Random Forest trained successfully")
        return model
    
    def train_xgboost(self, X_train, y_train):
        """Train XGBoost classifier."""
        logger.info("Training XGBoost...")
        
        model = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1]),
            random_state=self.random_state,
            eval_metric='logloss',
            use_label_encoder=False
        )
        
        model.fit(X_train, y_train)
        self.models['xgboost'] = model
        
        logger.info("XGBoost trained successfully")
        return model
    
    def train_lightgbm(self, X_train, y_train):
        """Train LightGBM classifier."""
        logger.info("Training LightGBM...")
        
        model = LGBMClassifier(
            n_estimators=100,
            max_depth=7,
            learning_rate=0.1,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight='balanced',
            random_state=self.random_state,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        self.models['lightgbm'] = model
        
        logger.info("LightGBM trained successfully")
        return model
    
    def evaluate_model(self, model, X_test, y_test, model_name):
        """Evaluate model performance."""
        logger.info(f"Evaluating {model_name}...")
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'average_precision': average_precision_score(y_test, y_pred_proba),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
        self.metrics[model_name] = metrics
        
        # Log metrics
        logger.info(f"{model_name} Metrics:")
        for metric, value in metrics.items():
            if metric != 'confusion_matrix':
                logger.info(f"  {metric}: {value:.4f}")
        
        logger.info(f"Confusion Matrix:\n{metrics['confusion_matrix']}")
        
        return metrics
    
    def cross_validate(self, model, X, y, n_splits=5):
        """Perform stratified k-fold cross validation."""
        logger.info(f"Performing {n_splits}-fold cross-validation...")
        
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        
        # Define scoring metrics
        scoring = {
            'accuracy': 'accuracy',
            'precision': 'precision',
            'recall': 'recall',
            'f1': 'f1',
            'roc_auc': 'roc_auc',
            'average_precision': 'average_precision'
        }
        
        cv_results = {}
        for metric_name, scorer in scoring.items():
            scores = cross_val_score(model, X, y, cv=cv, scoring=scorer, n_jobs=-1)
            cv_results[metric_name] = {
                'mean': scores.mean(),
                'std': scores.std(),
                'scores': scores
            }
            logger.info(f"{metric_name}: {scores.mean():.4f} (+/- {scores.std():.4f})")
        
        return cv_results
    
    def select_best_model(self):
        """Select the best model based on performance metrics."""
        logger.info("Selecting best model...")
        
        if not self.metrics:
            raise ValueError("No models have been evaluated yet")
        
        # Use F1-score as primary metric (balanced for precision and recall)
        best_score = -1
        best_model_name = None
        
        for model_name, metrics in self.metrics.items():
            # Weighted score: 40% F1 + 30% ROC-AUC + 30% Average Precision
            weighted_score = (
                0.4 * metrics['f1_score'] +
                0.3 * metrics['roc_auc'] +
                0.3 * metrics['average_precision']
            )
            
            logger.info(f"{model_name} weighted score: {weighted_score:.4f}")
            
            if weighted_score > best_score:
                best_score = weighted_score
                best_model_name = model_name
        
        self.best_model = self.models[best_model_name]
        logger.info(f"Selected best model: {best_model_name} with score: {best_score:.4f}")
        
        return best_model_name, self.best_model
    
    def plot_confusion_matrices(self):
        """Plot confusion matrices for all models."""
        n_models = len(self.metrics)
        fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))
        
        if n_models == 1:
            axes = [axes]
        
        for idx, (model_name, metrics) in enumerate(self.metrics.items()):
            cm = metrics['confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx])
            axes[idx].set_title(f'{model_name} Confusion Matrix')
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_metrics_comparison(self):
        """Plot comparison of metrics across models."""
        metrics_df = pd.DataFrame(self.metrics).T
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'average_precision']
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics_to_plot):
            if metric in metrics_df.columns:
                metrics_df[metric].plot(kind='bar', ax=axes[idx], color='skyblue')
                axes[idx].set_title(f'{metric.replace("_", " ").title()}')
                axes[idx].set_ylabel('Score')
                axes[idx].tick_params(axis='x', rotation=45)
                axes[idx].set_ylim([0, 1])
        
        plt.tight_layout()
        plt.savefig('metrics_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_model(self, model, filename):
        """Save trained model to file."""
        joblib.dump(model, filename)
        logger.info(f"Model saved to {filename}")
    
    def load_model(self, filename):
        """Load trained model from file."""
        model = joblib.load(filename)
        logger.info(f"Model loaded from {filename}")
        return model