#!/usr/bin/env python3
"""
Main pipeline script for fraud detection system.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from src.data_preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.model_training import FraudDetectionModel
from src.utils import setup_logging, save_results, plot_class_distribution
import logging
import warnings
warnings.filterwarnings('ignore')

def run_ecommerce_pipeline():
    """Run pipeline for e-commerce fraud detection."""
    logger = logging.getLogger(__name__)
    logger.info("Starting e-commerce fraud detection pipeline...")
    
    # Initialize components
    preprocessor = DataPreprocessor()
    feature_engineer = FeatureEngineer()
    model_trainer = FraudDetectionModel()
    
    # Load data
    fraud_data, ip_country, _ = preprocessor.load_data(
        'data/raw/Fraud_Data.csv',
        'data/raw/IpAddress_to_Country.csv',
        'data/raw/creditcard.csv'
    )
    
    # Clean data
    fraud_clean = preprocessor.clean_fraud_data(fraud_data)
    
    # Map IP to country
    fraud_with_country = preprocessor.map_ip_to_country(fraud_clean, ip_country)
    
    # Create features
    fraud_features = feature_engineer.create_time_features(fraud_with_country)
    fraud_features = feature_engineer.create_transaction_features(fraud_features)
    fraud_features = feature_engineer.create_browser_source_features(fraud_features)
    
    # Define categorical and numerical columns
    categorical_cols = ['source', 'browser', 'sex', 'country']
    numerical_cols = ['purchase_value', 'age', 'time_since_signup', 
                     'transaction_velocity', 'user_total_transactions']
    
    # Add engineered numerical features
    numerical_cols += [col for col in fraud_features.columns 
                      if col.endswith('_rate') or col.endswith('_deviation') 
                      or 'purchase_' in col]
    
    # Encode categorical features
    fraud_encoded = feature_engineer.encode_categorical_features(fraud_features, categorical_cols)
    
    # Scale numerical features
    fraud_scaled = feature_engineer.scale_numerical_features(fraud_encoded, numerical_cols)
    
    # Handle missing values
    fraud_final = feature_engineer.handle_missing_values(fraud_scaled)
    
    # Prepare features and target
    X = fraud_final.drop(['user_id', 'signup_time', 'purchase_time', 'device_id', 
                         'ip_address', 'ip_int', 'class'], axis=1, errors='ignore')
    y = fraud_final['class']
    
    # Save processed data
    fraud_final.to_csv('data/processed/ecommerce_fraud_processed.csv', index=False)
    X.to_csv('data/processed/ecommerce_X.csv', index=False)
    y.to_csv('data/processed/ecommerce_y.csv', index=False)
    
    # Train and evaluate models
    X_train, X_test, y_train, y_test = model_trainer.prepare_data(X, y)
    
    # Train models
    lr_model = model_trainer.train_baseline(X_train, y_train)
    rf_model = model_trainer.train_random_forest(X_train, y_train)
    xgb_model = model_trainer.train_xgboost(X_train, y_train)
    lgb_model = model_trainer.train_lightgbm(X_train, y_train)
    
    # Evaluate models
    lr_metrics = model_trainer.evaluate_model(lr_model, X_test, y_test, 'logistic_regression')
    rf_metrics = model_trainer.evaluate_model(rf_model, X_test, y_test, 'random_forest')
    xgb_metrics = model_trainer.evaluate_model(xgb_model, X_test, y_test, 'xgboost')
    lgb_metrics = model_trainer.evaluate_model(lgb_model, X_test, y_test, 'lightgbm')
    
    # Cross-validation
    logger.info("Performing cross-validation for XGBoost...")
    cv_results = model_trainer.cross_validate(xgb_model, X, y)
    
    # Select best model
    best_model_name, best_model = model_trainer.select_best_model()
    
    # Save best model
    model_trainer.save_model(best_model, f'models/best_ecommerce_model.pkl')
    
    # Plot results
    model_trainer.plot_confusion_matrices()
    model_trainer.plot_metrics_comparison()
    
    # Save results
    results = {
        'data_stats': {
            'original_shape': fraud_data.shape,
            'processed_shape': fraud_final.shape,
            'class_distribution': y.value_counts().to_dict(),
            'fraud_percentage': (y.sum() / len(y) * 100)
        },
        'models_trained': list(model_trainer.models.keys()),
        'metrics': model_trainer.metrics,
        'best_model': best_model_name,
        'cross_validation': cv_results
    }
    
    save_results(results, 'results/ecommerce_results.json')
    
    logger.info("E-commerce pipeline completed successfully!")
    return results

def run_creditcard_pipeline():
    """Run pipeline for credit card fraud detection."""
    logger = logging.getLogger(__name__)
    logger.info("Starting credit card fraud detection pipeline...")
    
    # Initialize components
    preprocessor = DataPreprocessor()
    feature_engineer = FeatureEngineer()
    model_trainer = FraudDetectionModel()
    
    # Load data
    _, _, creditcard = preprocessor.load_data(
        'data/raw/Fraud_Data.csv',
        'data/raw/IpAddress_to_Country.csv',
        'data/raw/creditcard.csv'
    )
    
    # Clean data
    cc_clean = preprocessor.clean_creditcard_data(creditcard)
    
    # Create features
    cc_features = feature_engineer.prepare_creditcard_features(cc_clean)
    
    # Scale features
    numerical_cols = [col for col in cc_features.columns 
                     if col not in ['Time', 'Amount', 'Class'] and not col.startswith('V')]
    numerical_cols += ['Amount_log']
    
    cc_scaled = feature_engineer.scale_numerical_features(cc_features, numerical_cols)
    
    # Handle missing values
    cc_final = feature_engineer.handle_missing_values(cc_scaled)
    
    # Prepare features and target
    X = cc_final.drop(['Class', 'Time', 'Amount'], axis=1, errors='ignore')
    y = cc_final['Class']
    
    # Save processed data
    cc_final.to_csv('data/processed/creditcard_processed.csv', index=False)
    X.to_csv('data/processed/creditcard_X.csv', index=False)
    y.to_csv('data/processed/creditcard_y.csv', index=False)
    
    # Train and evaluate models
    X_train, X_test, y_train, y_test = model_trainer.prepare_data(X, y)
    
    # Train models
    lr_model = model_trainer.train_baseline(X_train, y_train)
    rf_model = model_trainer.train_random_forest(X_train, y_train)
    xgb_model = model_trainer.train_xgboost(X_train, y_train)
    
    # Evaluate models
    lr_metrics = model_trainer.evaluate_model(lr_model, X_test, y_test, 'logistic_regression')
    rf_metrics = model_trainer.evaluate_model(rf_model, X_test, y_test, 'random_forest')
    xgb_metrics = model_trainer.evaluate_model(xgb_model, X_test, y_test, 'xgboost')
    
    # Cross-validation
    logger.info("Performing cross-validation for XGBoost...")
    cv_results = model_trainer.cross_validate(xgb_model, X, y)
    
    # Select best model
    best_model_name, best_model = model_trainer.select_best_model()
    
    # Save best model
    model_trainer.save_model(best_model, f'models/best_creditcard_model.pkl')
    
    # Plot results
    model_trainer.plot_confusion_matrices()
    model_trainer.plot_metrics_comparison()
    
    # Save results
    results = {
        'data_stats': {
            'original_shape': creditcard.shape,
            'processed_shape': cc_final.shape,
            'class_distribution': y.value_counts().to_dict(),
            'fraud_percentage': (y.sum() / len(y) * 100)
        },
        'models_trained': list(model_trainer.models.keys()),
        'metrics': model_trainer.metrics,
        'best_model': best_model_name,
        'cross_validation': cv_results
    }
    
    save_results(results, 'results/creditcard_results.json')
    
    logger.info("Credit card pipeline completed successfully!")
    return results

def main():
    """Main pipeline execution."""
    # Setup logging first
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Create directories
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("FRAUD DETECTION PIPELINE STARTED")
    logger.info("=" * 60)
    
    try:
        # Run e-commerce pipeline
        ecommerce_results = run_ecommerce_pipeline()
        
        logger.info("\n" + "=" * 60)
        logger.info("E-COMMERCE PIPELINE COMPLETED")
        logger.info("=" * 60)
        
        # Run credit card pipeline
        creditcard_results = run_creditcard_pipeline()
        
        logger.info("\n" + "=" * 60)
        logger.info("CREDIT CARD PIPELINE COMPLETED")
        logger.info("=" * 60)
        
        # Compare results
        logger.info("\nCOMPARISON OF RESULTS:")
        logger.info("-" * 40)
        
        for dataset, results in [("E-commerce", ecommerce_results), 
                               ("Credit Card", creditcard_results)]:
            logger.info(f"\n{dataset}:")
            logger.info(f"  Best Model: {results['best_model']}")
            logger.info(f"  Fraud Percentage: {results['data_stats']['fraud_percentage']:.4f}%")
            
            if results['best_model'] in results['metrics']:
                metrics = results['metrics'][results['best_model']]
                logger.info(f"  F1-Score: {metrics['f1_score']:.4f}")
                logger.info(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
                logger.info(f"  Avg Precision: {metrics['average_precision']:.4f}")
        
        logger.info("\n" + "=" * 60)
        logger.info("FRAUD DETECTION PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()