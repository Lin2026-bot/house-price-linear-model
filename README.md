# house-price-linear-model

## 📌 Project Overview

This project investigates the predictive limits of linear models in house price estimation through rigorous feature engineering and structured model design.

In contrast to typical machine learning workflows that rely on complex nonlinear models, this study deliberately constrains the modeling framework to linear approaches (OLS, Ridge, Lasso, ElasticNet). The objective is to demonstrate that, with well-designed features, linear models can achieve competitive performance while maintaining full interpretability and transparency.

---

## 🎯 Objectives

* Develop a high-performing house price prediction model under linear modeling constraints
* Minimize prediction error measured by Mean Absolute Percentage Error (MAPE)
* Preserve interpretability for real-world decision-making
* Benchmark performance against a nonlinear model (XGBoost)

---

## 📊 Dataset & Preprocessing

### Data Cleaning

* Removed **4,598 duplicate observations (~50% of dataset)**
* Filtered unrealistic price values (< 10,000 or > 10,000,000)
* Eliminated logically inconsistent records (e.g., zero bedrooms/bathrooms)

Final dataset: **4,546 observations**

### Target Transformation

* Applied **log transformation** to price to address skewness and heteroscedasticity
* Improved linearity between predictors and target

---

## 🔍 Exploratory Data Analysis

Key findings from EDA:

* **Strong linear relationship** between living area and log(price)
* **High multicollinearity** among size-related variables
* **Significant location heterogeneity** across cities and zip codes
* **Nonlinear temporal effects** in property age
* Renovation data is highly skewed → treated as a binary indicator

These insights directly guided the feature engineering strategy.

---

## 🧠 Feature Engineering

Feature engineering is the core driver of model performance in this project.

### 1. Size & Spatial Structure

* Log transformations: `log_total_area`
* Ratio features: `area_per_room`, `living_lot_ratio`
* Space composition: `basement_ratio`, `above_ratio`

### 2. Property Lifecycle

* `house_age` and quadratic term `house_age²`
* Binary indicators: `new_house`, `old_house`

### 3. Quality & Premium Effects

* Interaction terms:

  * `living_condition`
  * `view_area`, `waterfront_area`
* Premium indicators:

  * `large_house`, `luxury_house`

### 4. Structural Synergies

* `bed_bath_interaction` to capture functional balance

### 5. Location Encoding (Critical Component)

* One-hot encoding for city (low cardinality)
* Target encoding for zip code:

  * Mean price (`log_zip_avg_price`)
  * Variance (`zip_price_std`)
  * Frequency (`zip_count`)
* **KNN-based feature (key innovation):**

  * `log_knn_avg_price` capturing local comparable pricing

📌 All distribution-dependent features were computed strictly on the training set to prevent data leakage.

---

## ⚙️ Modeling Approach

### Models Implemented

* Ordinary Least Squares (OLS)
* Ridge Regression
* Lasso Regression
* ElasticNet
* Multi-Spline Ridge (B-spline basis expansion)

### Training Strategy

* Train-test split: **80 / 20**
* Model selection via **5-fold cross-validation**
* Target variable: **log(price)**
* Predictions transformed back to original scale for evaluation

---

## 📈 Model Performance

| Model               | Test MAPE  |
| ------------------- | ---------- |
| OLS                 | **0.1600** |
| Ridge               | 0.1606     |
| Lasso               | 0.1663     |
| ElasticNet          | 0.1614     |
| Multi-Spline Ridge  | 0.1606     |
| XGBoost (benchmark) | **0.1564** |

### Key Result

Linear models achieve performance within **~2.25%** of a well-tuned XGBoost model.

This indicates that feature engineering successfully captured the dominant nonlinear patterns in the data.

---

## 🔎 Model Interpretation

Interpretability analysis based on Ridge coefficients reveals:

### Dominant Drivers

* **Location (strongest factor)**

  * `log_zip_avg_price` dominates all features

* **Property Size**

  * `log_total_area`, `sqft_living`

### Structural & Functional Effects

* Negative impact of imbalanced layouts (`bed_bath_interaction`)
* Bathrooms strongly correlated with usability

### Temporal Dynamics

* `house_age`: negative effect (depreciation)
* `house_age²`: positive → diminishing depreciation over time

### Spatial Insights

* Urban premium (e.g., Seattle)
* Suburban discount (e.g., Renton, Auburn)

---

## 📉 Residual Diagnostics

* Systematic underestimation for low-priced properties
* Stable and unbiased predictions in mid-to-high price ranges
* Strong anchoring effect from location-based features

---

## 💡 Key Takeaways

* Feature engineering can **effectively linearize complex relationships**
* Linear models can **closely match nonlinear benchmarks**
* Interpretability provides **actionable economic insights**
* Location and comparable pricing dominate housing valuation

---

## 🚀 How to Run

```bash id="runproj"
pip install -r requirements.txt
python src/train.py
```

---

## 📁 Project Structure

See repository structure for details on data, notebooks, and source code organization.

---

## 📄 Report

Full methodology and analysis are documented in:

📎 `report/Assignment1.pdf`
