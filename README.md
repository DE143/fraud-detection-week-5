Fraud Detection System
📊 Business Context & Problem Statement

Company: Adey Innovations Inc. (Financial Technology Sector)

Problem: Improving fraud detection accuracy for e-commerce transactions and bank credit transactions to prevent financial losses while maintaining customer trust.

Business Impact:

    Direct Financial Loss Prevention: Effective fraud detection prevents unauthorized transactions and reduces chargebacks.

    Customer Trust & Experience: Minimizing false positives (legitimate transactions flagged as fraud) is crucial for customer retention.

    Regulatory Compliance: Financial institutions require robust fraud detection systems to meet security standards.

    Operational Efficiency: Automated detection reduces manual review workload for fraud analysts.

Stakeholders:

    Fraud Analysts: Need interpretable models to investigate flagged transactions

    Product Managers: Require balance between security and user experience

    Customers: Seek seamless yet secure transaction experiences

    Compliance Officers: Need auditable decision-making processes

📁 Project Structure
```
fraud-detection/
├── .vscode/
│   └── settings.json
├── .github/
│   └── workflows/
│       └── unittests.yml
├── data/                           # Add this folder to .gitignore
│   ├── raw/                       # Original datasets (DO NOT COMMIT)
│   │   ├── Fraud_Data.csv
│   │   ├── IpAddress_to_Country.csv
│   │   └── creditcard.csv
│   └── processed/                 # Cleaned and feature-engineered data
├── notebooks/
│   ├── __init__.py
│   ├── 01-eda-fraud-data.ipynb    # EDA for e-commerce data
│   ├── 02-eda-creditcard.ipynb    # EDA for bank transaction data
│   ├── 03-feature-engineering.ipynb
│   ├── 04-modeling.ipynb
│   ├── 05-shap-explainability.ipynb
│   └── README.md
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py      # Data cleaning and feature engineering
│   ├── models.py                  # Model training and evaluation
│   └── utils.py                   # Helper functions
├── tests/
│   ├── __init__.py
│   ├── test_preprocessing.py
│   └── test_models.py
├── models/                        # Saved model artifacts
│   ├── fraud_detection_model.pkl
│   └── model_performance_report.txt
├── reports/                       # Analysis outputs and visualizations
│   ├── figures/
│   │   ├── feature_importance.png
│   │   ├── shap_summary.png
│   │   └── confusion_matrix.png
│   └── business_recommendations.md
├── scripts/
│   ├── __init__.py
│   ├── train_model.py            # Full training pipeline
│   ├── predict.py               # Make predictions on new data
│   └── README.md
├── requirements.txt
├── environment.yml               # Conda environment (optional)
├── setup.py
├── .gitignore
└── README.md

```

📂 Data Sources
1. E-commerce Transaction Data (Fraud_Data.csv)

    Source: Provided dataset for fraud detection analysis

    Size: ~ [Number of rows] transactions, [Number of columns] features

    Columns:

        user_id: Unique user identifier

        signup_time, purchase_time: Transaction timestamps

        purchase_value: Transaction amount in dollars

        device_id: Device identifier

        source: Acquisition channel (SEO, Ads, etc.)

        browser: Browser used

        sex, age: User demographics

        ip_address: User's IP address

        class: Target (1 = fraud, 0 = legitimate)

2. IP Address Mapping (IpAddress_to_Country.csv)

    Purpose: Geolocation mapping for fraud pattern analysis

    Columns: IP ranges and corresponding countries

3. Bank Transaction Data (creditcard.csv)

    Source: Publicly available credit card fraud dataset

    Size: 284,807 transactions, 31 features

    Note: Features V1-V28 are PCA-transformed for privacy

    Imbalance: ~0.172% fraud cases

🚀 Installation & Setup
Prerequisites

    Python 3.8+

    Git

    4GB+ RAM recommended

1. Clone Repository
```
git clone https://github.com/DE143/fraud-detection.git
cd fraud-detection
```
2. Create Virtual Environment
```
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# OR using conda
conda create -n fraud-detection python=3.9
conda activate fraud-detection
```
3. Install Dependencies
```
pip install -r requirements.txt
```
4. Set Up Data Directory
```
# Create data directories (gitignored)
mkdir -p data/raw data/processed
mkdir -p models reports/figures

# Place your dataset files in data/raw/
# Required files: Fraud_Data.csv, IpAddress_to_Country.csv, creditcard.csv
```
🏃‍♂️ End-to-End Execution
Option 1: Run Complete Pipeline (Recommended)
```
# Step 1: Preprocess data
python scripts/train_model.py --preprocess

# Step 2: Train models
python scripts/train_model.py --train

# Step 3: Generate reports
python scripts/train_model.py --report
```
Option 2: Run Individual Notebooks

Execute notebooks in order:
```
jupyter notebook notebooks/01-eda-fraud-data.ipynb
# Then continue with 02, 03, 04, 05 in sequence
```
Option 3: Use Provided Scripts
```
# Train on e-commerce data
python scripts/train_model.py --dataset ecommerce

# Train on credit card data
python scripts/train_model.py --dataset creditcard

# Train on both datasets
python scripts/train_model.py --dataset both
```
Expected Outputs

After successful execution:

    Models saved in: models/ directory

        random_forest_model.pkl

        xgb_model.pkl

        model_metrics.json

    Reports generated in: reports/ directory

        figures/: All visualizations (PNG files)

        business_recommendations.md: Actionable insights

        model_performance_summary.txt: Performance metrics

    Processed data in: data/processed/

        Cleaned datasets ready for analysis

📊 Model Training Details
Training Process
```
# Example training command with parameters
python scripts/train_model.py \
    --model xgboost \
    --test_size 0.2 \
    --use_smote \
    --random_state 42
 ```
    Hyperparameter Tuning

Models include grid search for:

    Random Forest: n_estimators, max_depth, min_samples_split

    XGBoost: learning_rate, max_depth, subsample

    Logistic Regression: C, penalty

Evaluation Metrics (Imbalanced Data Focus)

    Primary: AUC-PR (Precision-Recall Curve)

    Secondary: F1-Score, Precision, Recall

    Business-oriented: Cost-sensitive metrics with custom weights

🔧 Key Features & Functionality
1. Advanced Feature Engineering

    Time-based features: Transaction velocity, time since signup

    Geolocation: Country mapping from IP addresses

    Behavioral patterns: User transaction history aggregation

    Device fingerprinting: Device usage patterns

2. Class Imbalance Handling

    SMOTE (Synthetic Minority Oversampling)

    Class weighting in models

    Stratified sampling for train-test splits

3. Model Explainability

    SHAP values for feature importance

    Individual prediction explanations

    Global feature impact analysis

📝 Assumptions & Limitations
Assumptions

    Data Quality: Input data follows expected schema and formats

    Temporal Consistency: Fraud patterns remain stable during model deployment

    Feature Availability: All engineered features can be computed in production

    Resource Availability: Sufficient memory/CPU for model training

Limitations

    Cold Start Problem: New users with limited transaction history

    Evolving Fraud Patterns: Models may need retraining as fraud tactics change

    False Positives: Some legitimate transactions may be flagged (trade-off)

    Data Privacy: Limited interpretability of PCA-transformed features in credit card data

    Geographic Coverage: IP-to-country mapping may have inaccuracies

Known Issues

    High memory usage during SMOTE application on large datasets

    Long training time for hyperparameter tuning

    SHAP analysis computationally intensive for large datasets

📈 Performance Expectations
E-commerce Dataset

    Target AUC-PR: > 0.85

    Precision: > 0.75 (minimize false positives)

    Recall: > 0.80 (catch most fraud cases)

Credit Card Dataset

    Target AUC-PR: > 0.90

    Precision: > 0.80

    Recall: > 0.75

🧪 Testing

Run unit tests to ensure functionality:
```
# Run all tests
python -m pytest tests/ -v

# Run specific test module
python -m pytest tests/test_models.py -v

# Run with coverage report
python -m pytest tests/ --cov=src --cov-report=html
```
🤝 Contributing

    Fork the repository

    Create a feature branch (git checkout -b feature/AmazingFeature)

    Commit changes (git commit -m 'Add some AmazingFeature')

    Push to branch (git push origin feature/AmazingFeature)

    Open a Pull Request

📚 References & Resources

    Credit Card Fraud Detection Dataset

    SMOTE: Synthetic Minority Over-sampling Technique

    SHAP Documentation

    Fraud Detection Best Practices

📞 Support

For questions or issues:

    Check the Issues page

    Contact: [derese641735.ew@gmail.com]

 
📋 Requirements File Details

requirements.txt includes:

    Core ML: scikit-learn, xgboost, imbalanced-learn

    Visualization: matplotlib, seaborn, plotly

    Data Processing: pandas, numpy

    Explainability: shap

    Utilities: jupyter, notebook, ipykernel

Install all dependencies with:
```
pip install -r requirements.txt
```
Last Updated: December 2025
Version: 1.0.0
Status: Production Ready


