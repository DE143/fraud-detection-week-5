

"""
Data preprocessing module for fraud detection system.
"""
import pandas as pd
import numpy as np
from datetime import datetime
import ipaddress
from typing import Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Handles data cleaning, preprocessing, and integration."""
    
    def __init__(self):
        self.country_mapping = None
        
    def load_data(self, fraud_data_path: str, ip_country_path: str, 
                 creditcard_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load all datasets."""
        logger.info("Loading datasets...")
        
        fraud_data = pd.read_csv(fraud_data_path)
        ip_country = pd.read_csv(ip_country_path)
        creditcard = pd.read_csv(creditcard_path)
        
        logger.info(f"Fraud data shape: {fraud_data.shape}")
        logger.info(f"IP-Country data shape: {ip_country.shape}")
        logger.info(f"Credit card data shape: {creditcard.shape}")
        
        return fraud_data, ip_country, creditcard
    
    def clean_fraud_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean the e-commerce fraud dataset."""
        logger.info("Cleaning fraud data...")
        
        # Create a copy
        df_clean = df.copy()
        
        # Handle missing values
        missing_counts = df_clean.isnull().sum()
        if missing_counts.any():
            logger.warning(f"Missing values found: {missing_counts[missing_counts > 0]}")
            
            # For numeric columns, fill with median
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if df_clean[col].isnull().any():
                    df_clean[col] = df_clean[col].fillna(df_clean[col].median())
            
            # For categorical columns, fill with mode
            categorical_cols = df_clean.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if df_clean[col].isnull().any():
                    df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
        
        # Convert timestamps
        df_clean['signup_time'] = pd.to_datetime(df_clean['signup_time'])
        df_clean['purchase_time'] = pd.to_datetime(df_clean['purchase_time'])
        
        # Remove duplicates
        initial_shape = df_clean.shape
        df_clean = df_clean.drop_duplicates()
        duplicates_removed = initial_shape[0] - df_clean.shape[0]
        if duplicates_removed > 0:
            logger.info(f"Removed {duplicates_removed} duplicate rows")
        
        # Validate data types
        df_clean['age'] = pd.to_numeric(df_clean['age'], errors='coerce')
        df_clean['purchase_value'] = pd.to_numeric(df_clean['purchase_value'], errors='coerce')
        
        # Drop rows with invalid ages
        invalid_age = df_clean['age'].isnull() | (df_clean['age'] < 0) | (df_clean['age'] > 120)
        if invalid_age.any():
            logger.warning(f"Removing {invalid_age.sum()} rows with invalid ages")
            df_clean = df_clean[~invalid_age]
        
        logger.info(f"Cleaned fraud data shape: {df_clean.shape}")
        return df_clean
    
    def clean_creditcard_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean the credit card fraud dataset."""
        logger.info("Cleaning credit card data...")
        
        df_clean = df.copy()
        
        # Check for missing values
        missing_counts = df_clean.isnull().sum()
        if missing_counts.any():
            logger.warning(f"Missing values in credit card data: {missing_counts[missing_counts > 0]}")
            # Fill with median for numeric columns
            for col in df_clean.columns:
                if df_clean[col].isnull().any() and col != 'Class':
                    df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        
        # Remove duplicates
        initial_shape = df_clean.shape
        df_clean = df_clean.drop_duplicates()
        duplicates_removed = initial_shape[0] - df_clean.shape[0]
        if duplicates_removed > 0:
            logger.info(f"Removed {duplicates_removed} duplicate rows from credit card data")
        
        # Scale 'Amount' column (will be properly scaled later, but ensure it's numeric)
        df_clean['Amount'] = pd.to_numeric(df_clean['Amount'], errors='coerce')
        df_clean = df_clean.dropna(subset=['Amount'])
        
        logger.info(f"Cleaned credit card data shape: {df_clean.shape}")
        return df_clean
    
    def ip_to_int(self, ip: str) -> int:
        """Convert IP address to integer."""
        try:
            return int(ipaddress.IPv4Address(ip))
        except:
            return 0
    
    def map_ip_to_country(self, fraud_df: pd.DataFrame, ip_country_df: pd.DataFrame) -> pd.DataFrame:
        """Map IP addresses to countries using range-based lookup."""
        logger.info("Mapping IP addresses to countries...")
        
        # Convert IPs to integers
        fraud_df['ip_int'] = fraud_df['ip_address'].apply(self.ip_to_int)
        
        # Sort IP ranges for efficient lookup
        ip_country_df = ip_country_df.sort_values('lower_bound_ip_address')
        ip_country_df['lower_bound'] = ip_country_df['lower_bound_ip_address'].apply(self.ip_to_int)
        ip_country_df['upper_bound'] = ip_country_df['upper_bound_ip_address'].apply(self.ip_to_int)
        
        # Function to find country for IP
        def find_country(ip_int):
            mask = (ip_country_df['lower_bound'] <= ip_int) & (ip_country_df['upper_bound'] >= ip_int)
            matches = ip_country_df[mask]
            if len(matches) > 0:
                return matches.iloc[0]['country']
            return 'Unknown'
        
        # Apply mapping
        fraud_df['country'] = fraud_df['ip_int'].apply(find_country)
        
        # Store country mapping for later use
        self.country_mapping = fraud_df[['ip_address', 'country']].drop_duplicates()
        
        logger.info(f"Countries mapped. Unique countries: {fraud_df['country'].nunique()}")
        return fraud_df
    
    def analyze_class_distribution(self, df: pd.DataFrame, target_col: str) -> Dict:
        """Analyze class distribution."""
        class_counts = df[target_col].value_counts()
        class_ratio = class_counts[1] / class_counts[0] if 1 in class_counts else 0
        
        logger.info(f"Class distribution: {dict(class_counts)}")
        logger.info(f"Fraud ratio: {class_ratio:.6f}")
        logger.info(f"Fraud percentage: {(class_counts[1] / len(df) * 100):.4f}%")
        
        return {
            'total': len(df),
            'non_fraud': class_counts.get(0, 0),
            'fraud': class_counts.get(1, 0),
            'fraud_ratio': class_ratio,
            'fraud_percentage': (class_counts.get(1, 0) / len(df) * 100) if len(df) > 0 else 0
        }