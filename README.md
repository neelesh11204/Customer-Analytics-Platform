# Customer-Analytics-Platform
End-to-end customer analytics platform for customer segmentation, churn prediction and customer lifetime value estimation using machine learning.
Customer Analytics Platform
An end-to-end customer analytics pipeline built on the IBM Telco Customer Churn dataset. The project combines customer segmentation, churn prediction, and customer lifetime value (CLV) forecasting into a single workflow, and translates the model outputs into concrete, prioritized business actions.

Overview
Telecom companies sit on rich behavioral data but often struggle to turn it into action. This project builds a pipeline that:

Cleans and engineers features from raw customer billing/usage data.
Segments customers using RFM analysis and unsupervised clustering.
Predicts churn risk with several classification models.
Forecasts customer lifetime value with regression models.
Combines segment, churn risk, and CLV into a campaign priority score that tells the business exactly who to target and how.
All work is in ML_project.ipynb.

Dataset
Telco Customer Churn (IBM sample dataset), bundled here as archive.zip.

7,043 customers, 21 raw attributes (demographics, account info, services subscribed, billing).
Target variables: Churn (classification) and derived Estimated_CLV (regression).
Class balance: 5,174 non-churners vs. 1,869 churners (~26.5% churn rate).
Pipeline
1. Data Cleaning & Preprocessing
Loads the raw CSV, fixes types (e.g. TotalCharges to numeric), handles missing values, and extracts the dataset from the bundled zip archive.

2. Exploratory Data Analysis
Distribution analysis of tenure and spending, contract-type breakdowns, revenue contribution by contract, churn distribution, correlation heatmap, and time-series checks.

3. Feature Engineering
Builds a business-oriented feature set adapted from classic RFM analysis to a subscription context:

Feature	Description
Customer_Tenure	Proxy for recency/loyalty
Monthly_Spending, Monetary_Value	Recurring and total value
Average_Order_Value	Average transaction size
Total_Transactions / Purchase_Frequency	Engagement volume
Product_Diversity	Number of services subscribed ("stickiness")
Engagement_Frequency	Tenure vs. product diversity ratio
Average_Revenue_per_Month, Tenure_Buckets, Has_Tech_Support	Additional business-driven features
4. Customer Segmentation
RFM segmentation into actionable groups: Champions, Loyal Customers, Potential Loyalists, New Customers, At-Risk Customers, Lost Customers.
K-Means clustering, with optimal k selected via the Elbow Method and Silhouette Score (best k = 2), separating customers into an "Inactive" cluster and a "High Value" cluster.
Hierarchical clustering (Agglomerative) as a cross-check, with a dendrogram for visual validation.
PCA used for 2D visualization of cluster separation.
5. Churn Prediction
Five classifiers trained and compared on held-out test data:

Model	Accuracy	Precision	Recall	F1 Score	ROC AUC
LightGBM	0.802	0.669	0.499	0.571	0.840
XGBoost	0.804	0.686	0.480	0.565	0.843
Logistic Regression	0.791	0.654	0.450	0.533	0.842
Random Forest	0.767	0.574	0.458	0.510	0.802
Decision Tree	0.732	0.493	0.491	0.492	0.658
Best model: LightGBM (selected by F1-Score). Top predictive features: Monthly_Spending, Monetary_Value, Engagement_Frequency, Recency, Product_Diversity.

6. Customer Lifetime Value (CLV) Prediction
Three regressors trained on engineered features to predict Estimated_CLV:

Model	MAE	RMSE	R²
Random Forest Regressor	4,886.9	10,556.5	1.000
Gradient Boosting Regressor	22,316.3	40,622.3	0.998
Linear Regression	178,990.1	325,397.6	0.899
Best model: Random Forest Regressor. Customers are then bucketed into Low / Medium / High CLV segments.

7. Business Recommendation Engine
Combines RFM segment, churn probability, and predicted CLV into a rule-based recommendation and a Campaign_Priority label (Critical / High / Medium / Low), so marketing/retention budget can be pointed at the customers who matter most — e.g. high-CLV customers with high churn risk get flagged Critical for priority retention outreach, while low-value, low-risk customers are flagged Low for low-cost, automated touchpoints.

Key Business Insights
~26.5% churn rate — a significant revenue risk if left unaddressed.
Churn risk correlates strongly with lower engagement, lower product diversity, and recent drops in spending — not just demographics.
A small share of customers (the "High Value" cluster) drives a disproportionate share of revenue; retaining them is materially more valuable than acquiring new customers.
Combining churn risk with CLV avoids two common mistakes: over-spending retention budget on low-value customers, and under-investing in high-value customers who are quietly at risk.
Tech Stack
Data/ML: pandas, numpy, scikit-learn, XGBoost, LightGBM, lifetimes
Visualization: matplotlib, seaborn
Environment: Jupyter Notebook
Repository Structure
.
├── ML_project.ipynb   # Full pipeline: cleaning → EDA → features → segmentation → churn → CLV → recommendations
├── archive.zip        # Raw Telco Customer Churn dataset
└── README.md
Getting Started
pip install -r requirements.txt
jupyter notebook ML_project.ipynb
Run the notebook top to bottom — each stage saves its output CSV, which the next stage reads back in.

Possible Extensions
Hyperparameter tuning (grid/Bayesian search) for the churn and CLV models.
Survival-analysis-based CLV (BG/NBD + Gamma-Gamma via lifetimes) on real transactional history rather than a single-snapshot proxy.
Deploy the scoring pipeline as an API for real-time campaign targeting.
A/B test the recommendation engine's retention offers against a control group.
