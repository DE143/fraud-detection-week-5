"""
Tests for data preprocessing functions.
"""
import pytest
import pandas as pd
import numpy as np
from src.data_preprocessing import DataPreprocessor

@pytest.fixture
def sample_fraud_data():
    """Create sample fraud data for testing."""
    data = {
        'user_id': [1, 2, 3, 4, 5],
        'signup_time': ['2023-01-01 10:00:00', '2023-01-02 11:00:00', 
                       '2023-01-03 12:00:00', '2023-01-04 13:00:00', 
                       '2023-01-05 14:00:00'],
        'purchase_time': ['2023-01-01 10:30:00', '2023-01-02 11:30:00',
                         '2023-01-03 12:30:00', '2023-01-04 13:30:00',
                         '2023-01-05 14:30:00'],
        'purchase_value': [100.0, 200.0, 150.0, 300.0, 250.0],
        'device_id': ['d1', 'd2', 'd3', 'd4', 'd5'],
        'source': ['SEO', 'Ads', 'Direct', 'SEO', 'Ads'],
        'browser': ['Chrome', 'Safari', 'Chrome', 'Firefox', 'Safari'],
        'sex': ['M', 'F', 'M', 'F', 'M'],
        'age': [25, 30, 35, 40, 45],
        'ip_address': ['192.168.1.1', '192.168.1.2', '192.168.1.3',
                      '192.168.1.4', '192.168.1.5'],
        'class': [0, 0, 1, 0, 1]
    }
    return pd.DataFrame(data)

def test_clean_fraud_data(sample_fraud_data):
    """Test data cleaning function."""
    preprocessor = DataPreprocessor()
    cleaned_data = preprocessor.clean_fraud_data(sample_fraud_data)
    
    # Check data types
    assert pd.api.types.is_datetime64_any_dtype(cleaned_data['signup_time'])
    assert pd.api.types.is_datetime64_any_dtype(cleaned_data['purchase_time'])
    
    # Check no missing values
    assert cleaned_data.isnull().sum().sum() == 0
    
    # Check shape (should be same if no duplicates)
    assert cleaned_data.shape == sample_fraud_data.shape

def test_ip_to_int():
    """Test IP address to integer conversion."""
    preprocessor = DataPreprocessor()
    
    # Test valid IP
    ip_int = preprocessor.ip_to_int('192.168.1.1')
    assert isinstance(ip_int, int)
    assert ip_int > 0
    
    # Test invalid IP
    ip_int = preprocessor.ip_to_int('invalid')
    assert ip_int == 0

def test_analyze_class_distribution(sample_fraud_data):
    """Test class distribution analysis."""
    preprocessor = DataPreprocessor()
    distribution = preprocessor.analyze_class_distribution(sample_fraud_data, 'class')
    
    assert 'total' in distribution
    assert 'non_fraud' in distribution
    assert 'fraud' in distribution
    assert distribution['total'] == 5
    assert distribution['fraud'] == 2
    assert distribution['non_fraud'] == 3