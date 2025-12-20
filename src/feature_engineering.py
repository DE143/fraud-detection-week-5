"""
Feature engineering module for fraud detection.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
import logging

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Creates and transforms features for fraud detection."""
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        
    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features."""
        logger.info("Creating time-based features...")
        
        df_features = df.copy()
        
        # Extract time components
        df_features['purchase_hour'] = df_features['purchase_time'].dt.hour
        df_features['purchase_dayofweek'] = df_features['purchase_time'].dt.dayofweek
        df_features['purchase_month'] = df_features['purchase_time'].dt.month
        
        # Time since signup
        df_features['time_since_signup'] = (
            df_features['purchase_time'] - df_features['signup_time']
        ).dt.total_seconds() / 3600  # in hours
        
        # Is business hours (9 AM to 5 PM)
        df_features['is_business_hours'] = ((df_features['purchase_hour'] >= 9) & 
                                           (df_features['purchase_hour'] <= 17)).astype(int)
        
        # Is weekend
        df_features['is_weekend'] = (df_features['purchase_dayofweek'] >= 5).astype(int)
        
        # Signup hour
        df_features['signup_hour'] = df_features['signup_time'].dt.hour
        
        logger.info(f"Created {len([c for c in df_features.columns if c not in df.columns])} time features")
        return df_features
    
    def create_transaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create transaction pattern features."""
        logger.info("Creating transaction pattern features...")
        
        df_features = df.copy()
        
        # Calculate transaction frequency per user
        user_transaction_counts = df_features.groupby('user_id').size().reset_index(name='user_total_transactions')
        df_features = pd.merge(df_features, user_transaction_counts, on='user_id', how='left')
        
        # Calculate transaction velocity (transactions per hour since signup)
        df_features['transaction_velocity'] = df_features['user_total_transactions'] / (df_features['time_since_signup'] + 1)
        
        # Device usage patterns
        device_user_counts = df_features.groupby('device_id')['user_id'].nunique().reset_index(name='unique_users_per_device')
        df_features = pd.merge(df_features, device_user_counts, on='device_id', how='left')
        
        # IP usage patterns
        ip_user_counts = df_features.groupby('ip_address')['user_id'].nunique().reset_index(name='unique_users_per_ip')
        df_features = pd.merge(df_features, ip_user_counts, on='ip_address', how='left')
        
        # Country-based features
        country_fraud_rate = df_features.groupby('country')['class'].mean().reset_index(name='country_fraud_rate')
        df_features = pd.merge(df_features, country_fraud_rate, on='country', how='left')
        
        # Purchase value statistics
        user_purchase_stats = df_features.groupby('user_id')['purchase_value'].agg([
            'mean', 'std', 'max', 'min'
        ]).reset_index()
        user_purchase_stats.columns = ['user_id', 'user_purchase_mean', 'user_purchase_std', 
                                      'user_purchase_max', 'user_purchase_min']
        df_features = pd.merge(df_features, user_purchase_stats, on='user_id', how='left')
        
        # Purchase value deviation from user mean
        df_features['purchase_value_deviation'] = (
            df_features['purchase_value'] - df_features['user_purchase_mean']
        ) / (df_features['user_purchase_std'] + 1e-6)
        
        logger.info(f"Created {len([c for c in df_features.columns if c not in df.columns])} transaction features")
        return df_features
    
    def create_browser_source_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features from browser and source columns."""
        logger.info("Creating browser and source features...")
        
        df_features = df.copy()
        
        # Browser fraud rate
        browser_fraud_rate = df_features.groupby('browser')['class'].mean().reset_index(name='browser_fraud_rate')
        df_features = pd.merge(df_features, browser_fraud_rate, on='browser', how='left')
        
        # Source fraud rate
        source_fraud_rate = df_features.groupby('source')['class'].mean().reset_index(name='source_fraud_rate')
        df_features = pd.merge(df_features, source_fraud_rate, on='source', how='left')
        
        # Browser-source combination
        df_features['browser_source'] = df_features['browser'] + '_' + df_features['source']
        browser_source_fraud_rate = df_features.groupby('browser_source')['class'].mean().reset_index(name='browser_source_fraud_rate')
        df_features = pd.merge(df_features, browser_source_fraud_rate, on='browser_source', how='left')
        
        # Drop intermediate column
        df_features = df_features.drop('browser_source', axis=1)
        
        logger.info(f"Created browser/source features")
        return df_features
    
    def encode_categorical_features(self, df: pd.DataFrame, categorical_cols: list) -> pd.DataFrame:
        """Encode categorical features."""
        logger.info("Encoding categorical features...")
        
        df_encoded = df.copy()
        
        for col in categorical_cols:
            if col in df_encoded.columns:
                # Use label encoding for high cardinality, one-hot for low cardinality
                if df_encoded[col].nunique() > 10:  # High cardinality
                    le = LabelEncoder()
                    df_encoded[f'{col}_encoded'] = le.fit_transform(df_encoded[col].astype(str))
                    self.encoders[col] = le
                else:  # Low cardinality - one-hot encode
                    dummies = pd.get_dummies(df_encoded[col], prefix=col, drop_first=True)
                    df_encoded = pd.concat([df_encoded, dummies], axis=1)
        
        # Drop original categorical columns
        df_encoded = df_encoded.drop(categorical_cols, axis=1, errors='ignore')
        
        logger.info(f"Encoded {len(categorical_cols)} categorical features")
        return df_encoded
    
    def scale_numerical_features(self, df: pd.DataFrame, numerical_cols: list) -> pd.DataFrame:
        """Scale numerical features."""
        logger.info("Scaling numerical features...")
        
        df_scaled = df.copy()
        
        for col in numerical_cols:
            if col in df_scaled.columns:
                scaler = StandardScaler()
                df_scaled[col] = scaler.fit_transform(df_scaled[[col]])
                self.scalers[col] = scaler
        
        logger.info(f"Scaled {len(numerical_cols)} numerical features")
        return df_scaled
    
    def prepare_creditcard_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for credit card dataset."""
        logger.info("Preparing credit card features...")
        
        df_features = df.copy()
        
        # Create time-based features
        df_features['hour_of_day'] = (df_features['Time'] // 3600) % 24
        df_features['day_of_week'] = (df_features['Time'] // (3600 * 24)) % 7
        
        # Log transform amount to handle skewness
        df_features['Amount_log'] = np.log1p(df_features['Amount'])
        
        # Interaction features
        df_features['V1_Amount'] = df_features['V1'] * df_features['Amount_log']
        df_features['V2_Amount'] = df_features['V2'] * df_features['Amount_log']
        df_features['V3_Amount'] = df_features['V3'] * df_features['Amount_log']
        
        # Statistical features
        v_cols = [f'V{i}' for i in range(1, 29)]
        df_features['V_mean'] = df_features[v_cols].mean(axis=1)
        df_features['V_std'] = df_features[v_cols].std(axis=1)
        df_features['V_max'] = df_features[v_cols].max(axis=1)
        df_features['V_min'] = df_features[v_cols].min(axis=1)
        
        logger.info(f"Created {len([c for c in df_features.columns if c not in df.columns])} credit card features")
        return df_features
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the feature set."""
        logger.info("Handling missing values...")
        
        df_clean = df.copy()
        
        # Fill numeric columns with median
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df_clean[col].isnull().any():
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        
        # Fill categorical columns with mode
        categorical_cols = df_clean.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df_clean[col].isnull().any():
                df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
        
        # Drop columns with too many missing values
        missing_ratio = df_clean.isnull().sum() / len(df_clean)
        columns_to_drop = missing_ratio[missing_ratio > 0.5].index.tolist()
        if columns_to_drop:
            logger.warning(f"Dropping columns with >50% missing values: {columns_to_drop}")
            df_clean = df_clean.drop(columns_to_drop, axis=1)
        
        # Drop any remaining rows with missing values
        initial_shape = df_clean.shape
        df_clean = df_clean.dropna()
        rows_dropped = initial_shape[0] - df_clean.shape[0]
        if rows_dropped > 0:
            logger.warning(f"Dropped {rows_dropped} rows with remaining missing values")
        
        logger.info(f"Final shape after handling missing values: {df_clean.shape}")
        return df_clean