"""
Utility functions for fraud detection system.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any
import logging
import json
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def setup_logging(log_file='fraud_detection.log'):
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def save_results(results: Dict[str, Any], filename: str):
    """Save results to JSON file."""
    with open(filename, 'w') as f:
        json.dump(results, f, indent=4, default=str)

def load_results(filename: str) -> Dict[str, Any]:
    """Load results from JSON file."""
    with open(filename, 'r') as f:
        return json.load(f)

def plot_class_distribution(y: pd.Series, title: str = 'Class Distribution'):
    """Plot class distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Count plot
    counts = y.value_counts()
    axes[0].bar(['Non-Fraud', 'Fraud'], counts.values, color=['blue', 'red'])
    axes[0].set_title(f'{title} - Count')
    axes[0].set_ylabel('Count')
    
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + max(counts.values)*0.01, str(v), ha='center')
    
    # Percentage plot
    percentages = counts / counts.sum() * 100
    axes[1].bar(['Non-Fraud', 'Fraud'], percentages.values, color=['blue', 'red'])
    axes[1].set_title(f'{title} - Percentage')
    axes[1].set_ylabel('Percentage (%)')
    
    for i, v in enumerate(percentages.values):
        axes[1].text(i, v + 1, f'{v:.2f}%', ha='center')
    
    plt.tight_layout()
    plt.savefig('class_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return counts, percentages

def plot_feature_distributions(df: pd.DataFrame, features: List[str], n_cols: int = 4):
    """Plot distributions of selected features."""
    n_features = len(features)
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
    axes = axes.flatten()
    
    for idx, feature in enumerate(features):
        if feature in df.columns:
            if df[feature].dtype in ['int64', 'float64']:
                # Histogram for numerical features
                axes[idx].hist(df[feature].dropna(), bins=50, alpha=0.7, edgecolor='black')
                axes[idx].set_title(f'{feature} Distribution')
                axes[idx].set_xlabel(feature)
                axes[idx].set_ylabel('Frequency')
            else:
                # Bar plot for categorical features
                value_counts = df[feature].value_counts().head(10)
                axes[idx].bar(value_counts.index.astype(str), value_counts.values)
                axes[idx].set_title(f'{feature} Top 10 Values')
                axes[idx].set_xlabel(feature)
                axes[idx].set_ylabel('Count')
                axes[idx].tick_params(axis='x', rotation=45)
    
    # Hide empty subplots
    for idx in range(n_features, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('feature_distributions.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_correlation_matrix(df: pd.DataFrame, method: str = 'pearson'):
    """Plot correlation matrix for numerical features."""
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numerical_cols) > 1:
        corr_matrix = df[numerical_cols].corr(method=method)
        
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        
        sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                   square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=False)
        
        plt.title(f'Correlation Matrix ({method.capitalize()})', fontsize=16)
        plt.tight_layout()
        plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return corr_matrix
    return None

def plot_fraud_by_feature(df: pd.DataFrame, feature: str, target: str = 'class'):
    """Plot fraud rate by feature values."""
    if feature in df.columns and target in df.columns:
        # For numerical features, create bins
        if df[feature].dtype in ['int64', 'float64']:
            df_copy = df.copy()
            df_copy['binned'] = pd.qcut(df_copy[feature], q=10, duplicates='drop')
            fraud_rates = df_copy.groupby('binned')[target].mean()
            x_labels = [str(interval) for interval in fraud_rates.index]
        else:
            # For categorical features, take top 20 values
            top_values = df[feature].value_counts().head(20).index
            df_filtered = df[df[feature].isin(top_values)]
            fraud_rates = df_filtered.groupby(feature)[target].mean().sort_values(ascending=False)
            x_labels = fraud_rates.index.astype(str)
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(range(len(fraud_rates)), fraud_rates.values)
        plt.title(f'Fraud Rate by {feature}', fontsize=14)
        plt.xlabel(feature)
        plt.ylabel('Fraud Rate')
        plt.xticks(range(len(fraud_rates)), x_labels, rotation=45, ha='right')
        
        # Color bars by fraud rate
        for bar, rate in zip(bars, fraud_rates.values):
            bar.set_color(plt.cm.RdYlGn_r(rate))  # Red for high fraud, green for low
        
        plt.tight_layout()
        plt.savefig(f'fraud_rate_by_{feature}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return fraud_rates
    return None