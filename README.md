# Vendor Performance Data Analysis

An end-to-end data analytics and machine learning project I built to analyze vendor and brand performance using sales, purchasing, pricing, inventory, and freight data.

The project combines SQL-based data aggregation, data cleaning, exploratory data analysis, visualization, feature engineering, and machine learning to answer practical business questions such as:

- Which vendors and brands contribute the most to sales and profit?
- Which vendors or brands are underperforming?
- How does pricing affect profitability?
- How efficiently is inventory being turned over?
- Can profit margin be predicted from vendor and transaction-level characteristics?
- Can vendors be classified into High, Medium, and Low performance categories?
- Can potentially risky or anomalous vendors be detected automatically?

## Project Objective

Effective vendor and inventory management is important for maintaining profitability and reducing unnecessary costs.

In this project, I analyze vendor performance from multiple perspectives, including:

- Sales performance
- Purchase performance
- Profitability
- Pricing and markup
- Inventory turnover
- Freight cost
- Vendor and brand contribution
- Performance classification
- Risk and anomaly detection

My overall goal is to transform raw transactional data into a structured analytical dataset and then use statistical analysis and machine learning to generate actionable vendor-level insights.

## Business Problems Addressed

My analysis is designed to:

1. Identify underperforming brands that may require pricing or promotional adjustments.
2. Identify vendors contributing significantly to sales and gross profit.
3. Analyze the relationship between purchasing volume and unit costs.
4. Evaluate inventory turnover and identify inefficient inventory movement.
5. Examine profitability differences between high-performing and low-performing vendors.
6. Predict expected profit margin for a vendor-brand combination.
7. Classify vendors into High, Medium, and Low performance categories.
8. Detect unusual or potentially risky vendor records using rule-based and anomaly-detection methods.

## Project Pipeline

The project follows a complete data-to-insight pipeline:

```text
Raw CSV Data
     │
     ▼
MySQL Database
     │
     ▼
Data Ingestion
     │
     ▼
SQL Aggregation + ETL
     │
     ▼
vendor_sales_summary
     │
     ├──────────────► Exploratory Data Analysis
     │
     ├──────────────► Visualization & Business Insights
     │
     └──────────────► Machine Learning
                          │
                          ├── Profit Margin Prediction
                          ├── Vendor Performance Classification
                          └── Risk / Anomaly Detection
                                      │
                                      ▼
                              Saved ML Models
                                      │
                                      ▼
                              Streamlit Application
```

Execution Order:

```text
Raw Data
   ↓
Load CSV files into MySQL
   ↓
Run EDAscript.py
   ↓
Generate vendor_sales_summary
   ↓
Run EDA.ipynb
   ↓
Run Anal_Viz.ipynb
   ↓
Run ML_Models.ipynb
   ↓
Save trained model artifacts
   ↓
Run Streamlit application
```

---

## 1. Data Sources

The project starts with multiple raw CSV files containing information related to:

- Purchases
- Sales
- Purchase prices
- Vendor invoices
- Freight
- Vendor and brand information

I load the raw files into MySQL so that the project can perform structured SQL-based aggregation and joins before the analytical stage.

## 2. Database Ingestion

### `ingestion_db.py`

The ingestion script loads the raw CSV files from the dataset directory and creates corresponding MySQL tables.

Each CSV file is read using Pandas and written to MySQL using SQLAlchemy.

Conceptually:

```text
CSV File
    ↓
Pandas DataFrame
    ↓
SQLAlchemy
    ↓
MySQL Table
```

The script uses `to_sql()` with `if_exists="replace"`, meaning the corresponding table is recreated when the ingestion process is rerun.
This makes the ingestion process useful when rebuilding the database from the original raw dataset.

## 3. Database Connection

### `engine.py`

`engine.py` creates the SQLAlchemy connection used throughout the project.

Database credentials are loaded from environment variables using `python-dotenv`.

The connection uses:

```text
MySQL
   ↓
SQLAlchemy
   ↓
Pandas / Python scripts
```

The current implementation reads:

```text
DB_username
DB_password
DB_host
DB_database
```

from the environment and constructs a MySQL connection using the PyMySQL driver.

For security, database credentials should not be hard-coded into the source code.

## 4. ETL and Data Preparation

### `EDAscript.py`

`EDAscript.py` is the main ETL/preparation script of the project.

It performs three major tasks:

1. Extracts and aggregates information from the raw MySQL tables.
2. Cleans the resulting dataset and engineers analytical metrics.
3. Stores the processed dataset as `vendor_sales_summary`.

The script uses SQL Common Table Expressions (CTEs) to separately aggregate:

- Freight information
- Purchase information
- Sales information

These summaries are then joined using vendor and brand identifiers.

### Vendor Sales Summary

The resulting analytical table contains information such as:

- VendorNumber
- VendorName
- Brand
- Description
- PurchasePrice
- Volume
- ActualPrice
- TotalPurchaseQuantity
- TotalPurchaseDollars
- TotalSalesQuantity
- TotalSalesDollars
- TotalSalesPrice
- TotalExciseTax
- TotalFreightCost

This pre-aggregated table avoids repeatedly performing expensive joins and aggregations during analysis or dashboarding.

## 5. Data Cleaning

The ETL process performs several cleaning operations.

### Missing Values

Missing values are filled with zero:

```python
df.fillna(0, inplace=True)
```

### Vendor Name Standardization

Leading and trailing whitespace is removed from vendor names:

```python
df["VendorName"] = df["VendorName"].str.strip()
```

### Numeric Conversion

The `Volume` column is explicitly converted to a floating-point type.

### Infinite Values

After feature engineering, infinite values are replaced with missing values before the processed data is written back to MySQL.

These operations are implemented directly in the ETL workflow.

## 6. Feature Engineering

Several important business metrics are derived from the aggregated data.

### Gross Profit

```text
GrossProfit =
TotalSalesDollars - TotalPurchaseDollars
```

This represents the difference between sales revenue and purchase cost.

### Profit Margin

```text
ProfitMargin =
(GrossProfit / TotalSalesDollars) × 100
```

Profit margin is later used as the primary regression target.

### Stock Turnover

```text
StockTurnover =
TotalSalesQuantity / TotalPurchaseQuantity
```

This provides an indication of how efficiently purchased inventory is being sold.

### Sales-to-Purchase Ratio

```text
SalestoPurchaseRatio =
TotalSalesDollars / TotalPurchaseDollars
```

This provides a simple comparison between sales value and purchasing cost.

These metrics are generated during the ETL stage and become the foundation for subsequent analysis.

## 7. Exploratory Data Analysis

### `EDA.ipynb`

I use the EDA notebook to understand the structure and statistical characteristics of the processed vendor dataset before applying machine learning.

The analysis focuses on:

- Dataset structure
- Data types
- Missing values
- Descriptive statistics
- Distributions
- Outliers
- Profitability
- Vendor performance
- Inventory turnover
- Relationships between numerical variables
- Correlations between business metrics

The EDA stage is important because the machine learning pipeline is based on observations made during data exploration.

For example, the ML notebook starts with 10,648 records and subsequently performs cleaning and filtering before model development.

## 8. Visualization and Business Analysis

### `Anal_Viz.ipynb`

The visualization notebook focuses on turning the cleaned analytical dataset into business-oriented insights.

The analysis examines areas such as:

- Vendor sales contribution
- Purchase contribution
- Gross profit
- Profit margin
- Inventory turnover
- Pricing relationships
- Vendor-level comparisons
- Brand-level performance
- Freight and cost-related patterns

I also use correlation analysis to understand relationships between important numerical variables.

For example, the Streamlit application I built examines relationships among:

```text
PurchasePrice
ActualPrice
Volume
TotalSalesDollars
TotalPurchaseDollars
GrossProfit
ProfitMargin
StockTurnover
SalestoPurchaseRatio
TotalFreightCost
```

I specifically observed the strong relationship between `SalestoPurchaseRatio` and `ProfitMargin`, which is relevant when selecting features for machine learning.

## 9. Machine Learning

### `ML_Models.ipynb`

The project contains three machine learning tasks:

| Task                              | Type              | Algorithm                     |
| --------------------------------- | ----------------- | ----------------------------- |
| Profit Margin Prediction          | Regression        | XGBoost                       |
| Vendor Performance Classification | Classification    | XGBoost                       |
| Risk Detection                    | Anomaly Detection | Rule-based + Isolation Forest |

The models use `random_state=42` for reproducibility.

## 10. Machine Learning Data Preparation

Before training the models, the ML notebook performs additional preprocessing.

### Removing Zero-Sales Records

Records where:

```text
TotalSalesDollars <= 0
```

are removed.

My reasoning is that a vendor-brand record with no sales does not provide useful information for the profit-margin prediction task.

### Removing Extreme Profit Margin Values

Records with:

```text
ProfitMargin <= -200
```

are removed.

I treated these extreme values as data artifacts during the analysis.

### Capping Stock Turnover

Stock turnover is capped at its 99th percentile to reduce the influence of extremely high turnover ratios caused by very small purchase quantities.

### Additional Features

I create two additional features:

#### PriceMarkup

```text
PriceMarkup =
(ActualPrice - PurchasePrice) / PurchasePrice
```

This represents the relative markup between purchase and selling price.

#### OrderSize

`OrderSize` divides purchase quantity into four quartile-based groups:

```text
0 → smallest purchase quantities
1 → lower-middle quantities
2 → upper-middle quantities
3 → largest quantities
```

After preprocessing, the dataset contains:

```text
10,019 records
20 columns
```

and the supervised models use a nine-feature input matrix.

## 11. Feature Selection

I use the same nine features for both supervised learning tasks.

```text
PurchasePrice
ActualPrice
Volume
TotalPurchaseQuantity
TotalSalesQuantity
TotalPurchaseDollars
TotalSalesDollars
PriceMarkup
OrderSize
```

The feature matrix therefore has:

```text
10,019 rows × 9 features
```

I intentionally exclude the following variables:

```text
GrossProfit
ProfitMargin
StockTurnover
```

I do this to reduce target leakage because `ProfitMargin` is the regression target, while `GrossProfit` is directly involved in calculating it.
`StockTurnover` is also excluded from the supervised feature set because it is used in the performance-label construction and risk analysis.

## 12. Model 1 — Profit Margin Prediction

### Objective

Predict the expected `ProfitMargin` of a vendor-brand record from purchasing, pricing, sales, and volume-related features.

### Algorithm

```text
XGBoost Regressor
```

### Train/Test Split

I divide the data using:

```text
80% → Training
20% → Testing
```

with:

```text
random_state = 42
```

The resulting split is:

```text
Training samples: 8,015
Testing samples: 2,004
```

### Model Configuration

```python
XGBRegressor(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    random_state=42,
    verbosity=0
)
```

### Evaluation Metrics

I evaluate the model using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

### Current Test Results

```text
MAE  = 6.17
RMSE = 9.40
R²   = 0.9440
```

The model therefore explains approximately 94.4% of the variance in the held-out test target according to the notebook's R² calculation.

## 13. Model 2 — Vendor Performance Classification

### Objective

Classify vendor-brand records into:

```text
High
Medium
Low
```

performance categories.

### Label Construction

The performance label is not taken directly from an existing column.

Instead:

1. I percentile-rank vendors according to `ProfitMargin`.
2. I percentile-rank vendors according to `StockTurnover`.
3. I add the two percentile ranks.
4. I divide the resulting composite score into three groups.

Conceptually:

```text
Profit Margin Rank
        +
Stock Turnover Rank
        ↓
Composite Performance Score
        ↓
        ├── Low
        ├── Medium
        └── High
```

This produces a balanced three-class classification problem.

### Algorithm

```text
XGBoost Classifier
```

### Model Configuration

```python
XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    eval_metric="mlogloss",
    random_state=42,
    verbosity=0
)
```

The same nine-feature set used by the regression model is used here.

### Evaluation

I evaluate the classifier using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

### Current Test Results

```text
Accuracy: 0.78
Macro F1: 0.78
Weighted F1: 0.78
```

Per-class results from the current notebook:

| Class  | Precision | Recall | F1   |
| ------ | --------- | ------ | ---- |
| High   | 0.88      | 0.76   | 0.81 |
| Low    | 0.82      | 0.83   | 0.83 |
| Medium | 0.66      | 0.74   | 0.70 |

The test set contains 2,004 samples, with 668 samples in each class.

## 14. Model 3 — Risk and Anomaly Detection

Risk detection uses two complementary approaches:

1. Rule-Based Detection
2. Isolation Forest

This allows me to combine an easily interpretable business rule with an unsupervised anomaly-detection method.

### Rule-Based Risk Detection

A vendor is flagged when both conditions are satisfied:

```text
ProfitMargin < 25th percentile
AND
StockTurnover < 25th percentile
```

This provides a simple and business-readable definition of a potentially problematic vendor.

The thresholds are calculated from the processed dataset rather than manually hard-coded values.

### Isolation Forest

I use Isolation Forest to identify records that behave unusually across several financial and operational variables.

The anomaly feature set is:

```text
ProfitMargin
StockTurnover
GrossProfit
SalestoPurchaseRatio
```

Before training, these features are standardized using `StandardScaler`.

The Isolation Forest configuration is:

```python
IsolationForest(
    n_estimators=100,
    contamination=0.1,
    random_state=42
)
```

The model flags approximately 10% of the dataset as anomalous according to the configured contamination level.

Current notebook result:

```text
Flagged risky: 1,002 records
Percentage: 10.0%
```

An anomaly score is also generated to indicate how unusual a record is relative to the learned distribution.

### Combined Risk Analysis

My final risk analysis compares:

```text
Rule-Based Risk
        +
Isolation Forest Risk
```

This produces three useful groups:

```text
Flagged by both methods
        ↓
Rule-based only
        ↓
Isolation Forest only
```

The combination allows me to identify both clearly defined business risks and less obvious statistical anomalies.

## 15. Model Interpretability

The project includes feature-importance analysis for the XGBoost models.

For the regression model, feature importance is used to understand which input variables contribute most strongly to profit-margin predictions.

The classification model similarly provides feature importance for understanding which variables contribute to the High / Medium / Low performance predictions.

I also use:

- Actual vs Predicted plots
- Confusion matrices
- Per-class classification metrics
- Correlation heatmaps
- Distribution plots
- Risk scatter plots
- Anomaly-score distributions

These visualizations make the models easier to inspect rather than treating them as black boxes.

## 16. Model Artifacts

I export the trained models and supporting objects using `joblib`.

The ML pipeline generates:

```text
models/
│
├── features.pkl
├── anomaly_features.pkl
│
├── xgb_reg.pkl
├── reg_test_split.pkl
│
├── xgb_clf.pkl
├── label_encoder.pkl
├── clf_test_split.pkl
│
├── iso_forest.pkl
├── iso_scaler.pkl
└── iso_thresholds.pkl
```

These artifacts allow the application layer to use the already-trained models without retraining them every time.

The current notebook exports all 10 required artifacts in its final cell.

## 17. Streamlit Application

The Streamlit component is the final presentation layer of the project.

I intentionally kept it separate from the main data-analysis and model-training workflow.

The application provides four main sections:

```text
Data Overview
     ↓
Profit Margin Prediction
     ↓
Performance Classification
     ↓
Risk Detection
```

The application loads the previously trained models rather than training models at runtime.

The current application provides:

- Dataset overview
- Distribution visualizations
- Correlation analysis
- Profit-margin prediction
- Vendor performance classification
- Risk detection
- Interactive prediction for new vendor inputs

The Streamlit pages are therefore primarily intended to expose the results of the analysis and machine learning pipeline in an interactive form.

## 18. Project Structure

A simplified project structure is:

```text
Vendor Performance Data Analysis/
│
├── dataset/
│   └── data/
│       └── data/
│           ├── *.csv
│
├── engine.py
├── ingestion_db.py
├── EDAscript.py
│
├── EDA.ipynb
├── Anal_Viz.ipynb
├── ML_Models.ipynb
│
├── models/
│   ├── features.pkl
│   ├── anomaly_features.pkl
│   ├── xgb_reg.pkl
│   ├── reg_test_split.pkl
│   ├── xgb_clf.pkl
│   ├── label_encoder.pkl
│   ├── clf_test_split.pkl
│   ├── iso_forest.pkl
│   ├── iso_scaler.pkl
│   └── iso_thresholds.pkl
│
├── logs/
│
└── streamlit_vendor_app/
    └── streamlit_app/
        ├── app.py
        ├── utils.py
        ├── models/
        └── pages/
            ├── 1_Data_Overview.py
            ├── 2_Profit_Margin.py
            ├── 3_Performance_Class.py
            └── 4_Risk_Detection.py
```

## 19. Technologies Used

### Data Processing

- Python
- Pandas
- NumPy

### Database

- MySQL
- SQLAlchemy
- PyMySQL

### Data Analysis & Visualization

- Jupyter Notebook
- Matplotlib
- Seaborn
- Plotly

### Machine Learning

- Scikit-learn
- XGBoost
- Isolation Forest

### Model Persistence

- Joblib

### Application

- Streamlit

## 20. Installation

Clone the repository:

```bash
git clone <repository-url>
cd "Vendor Performance Data Analysis"
```

Create and activate a virtual environment:

```bash
python -m venv ml_env
```

Windows:

```bash
ml_env\Scripts\activate
```

Install the required packages:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost sqlalchemy pymysql python-dotenv joblib jupyter streamlit plotly
```

## 21. Environment Configuration

Create a `.env` file in the project root.

Example:

```env
DB_username=root
DB_password=your_mysql_password
DB_host=localhost
DB_database=vendor_analysis
```

Do not commit `.env` to GitHub.

Add it to `.gitignore`:

```text
.env
```

## 22. Running the Project from Scratch

If starting with the raw dataset, follow this order.

### Step 1 — Verify the raw CSV files

Make sure the raw dataset is present in the expected dataset directory.

### Step 2 — Start MySQL

Make sure MySQL is running and the target database exists.

### Step 3 — Load raw data

Run:

```bash
python ingestion_db.py
```

This loads the raw CSV files into MySQL tables.

### Step 4 — Run the ETL pipeline

Run:

```bash
python EDAscript.py
```

This:

- Reads the raw database tables
- Aggregates sales, purchases, and freight
- Joins the resulting summaries
- Cleans the data
- Creates business metrics
- Writes `vendor_sales_summary` back to MySQL

### Step 5 — Run EDA

Open:

```text
EDA.ipynb
```

and execute the notebook.

### Step 6 — Run visualization and business analysis

Open:

```text
Anal_Viz.ipynb
```

and execute the notebook.

### Step 7 — Train the machine learning models

Open:

```text
ML_Models.ipynb
```

Run the notebook from the beginning.

This is important because the notebook creates the cleaned dataset, feature matrix, labels, train/test splits, models, and supporting objects sequentially.

### Step 8 — Export model artifacts

Run the final model-export cell.

It creates:

```text
models/
```

containing the required `.pkl` files.

### Step 9 — Run the application

Place the generated model artifacts in the application's `models/` directory and run:

```bash
cd streamlit_vendor_app/streamlit_app
streamlit run app.py
```

## 23. Reproducibility

I designed the project so that the analytical and ML pipeline can be rebuilt when the underlying data changes.

If the dataset is modified or replaced, the recommended process is:

```text
New Raw Dataset
      ↓
ingestion_db.py
      ↓
EDAscript.py
      ↓
vendor_sales_summary
      ↓
EDA.ipynb
      ↓
Anal_Viz.ipynb
      ↓
ML_Models.ipynb
      ↓
New Model Artifacts
      ↓
Streamlit Application
```

The trained `.pkl` files should be regenerated whenever the dataset or model configuration changes.

## 24. Key Design Decisions

### Pre-aggregation

I create `vendor_sales_summary` instead of repeatedly performing large joins during analysis.

### Business-oriented feature engineering

Metrics such as:

```text
GrossProfit
ProfitMargin
StockTurnover
SalestoPurchaseRatio
```

translate raw transactional data into meaningful business indicators.

### Leakage prevention

Variables directly related to the target or label construction are excluded from the supervised feature set.

### Same feature set for supervised models

Both the regression and classification models use the same nine input features, keeping the modeling pipeline simple and consistent.

### Interpretable risk detection

Risk detection combines a transparent percentile-based rule with Isolation Forest so that both business-readable and statistical anomaly detection are available.

### Separate analysis and deployment

The notebooks are responsible for:

```text
Analysis → Modeling → Evaluation → Export
```

while Streamlit is responsible for:

```text
Loading → Visualization → Interactive Prediction
```

This keeps model training separate from the application layer.

## 25. Results Summary

| Component                  | Approach                      | Output                              |
| -------------------------- | ----------------------------- | ----------------------------------- |
| Data Preparation           | SQL + Pandas ETL              | Vendor-level analytical dataset     |
| Profitability Analysis     | Feature engineering + EDA     | Gross Profit, Profit Margin         |
| Inventory Analysis         | Stock Turnover                | Inventory efficiency insights       |
| Profit Margin Prediction   | XGBoost Regression            | Continuous profit-margin prediction |
| Performance Classification | XGBoost Classification        | High / Medium / Low                 |
| Risk Detection             | Rule + Isolation Forest       | Risk flags + anomaly scores         |
| Visualization              | Matplotlib / Seaborn / Plotly | Business and model insights         |
| Application                | Streamlit                     | Interactive analytical interface    |

## 26. Main Takeaways

This project demonstrates an end-to-end workflow rather than only training a machine learning model.

The complete process covers:

```text
Raw Data
   ↓
Database
   ↓
SQL Aggregation
   ↓
Data Cleaning
   ↓
Feature Engineering
   ↓
Exploratory Data Analysis
   ↓
Business Visualization
   ↓
Machine Learning
   ↓
Model Evaluation
   ↓
Model Persistence
   ↓
Interactive Application
```

The main machine learning components are:

```text
XGBoost Regression
        ↓
Profit Margin Prediction

XGBoost Classification
        ↓
Vendor Performance Classification

Rule-Based Detection + Isolation Forest
        ↓
Vendor Risk / Anomaly Detection
```

## 27. Future Improvements

Possible extensions I'm considering include:

- Hyperparameter tuning and cross-validation
- Additional vendor-level aggregation
- Time-based sales and profitability analysis
- Vendor trend analysis
- Automated model retraining
- Model explainability using SHAP
- More advanced anomaly-detection techniques
- Automated reporting
- API-based model serving
- Cloud deployment
- Scheduled data ingestion and retraining
