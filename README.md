# 📡 Telco Customer Churn Prediction: Optimizing Retention Strategy

![Python](https://img.shields.io/badge/Python-3.9-blue)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit_Learn-orange)
![Status](https://img.shields.io/badge/Status-Completed-green)

## 📌 Project Overview
In the telecommunications industry, acquiring a new customer is significantly more expensive than retaining an existing one. **Customer Churn** (when a customer stops doing business with a company) is a critical metric.

This project focuses on building an End-to-End Machine Learning model to **predict which customers are at risk of churning**. The goal is to provide actionable insights for the marketing team to launch proactive retention campaigns.

**Key Achievement:** Successfully improved the model's ability to detect churners (**Recall**) from **46% (Baseline)** to **68% (Final)** by implementing SMOTE and Hyperparameter Tuning.

---

## 💼 Business Understanding & Problem Statement
* **The Problem:** The dataset is **highly imbalanced** (loyal customers significantly outnumber churners). Standard models tend to be biased toward the majority class, failing to detect the actual customers who are leaving (High False Negatives).
* **The Goal:** Maximize **Recall** (Sensitivity). In this business context, a **False Negative** (failing to detect a churner) is more costly than a **False Positive** (wrongly flagging a loyal customer), as the company loses revenue when a customer leaves unnoticed.

---

## 🛠️ Methodology
The project followed a structured Data Science lifecycle:

1.  **Data Cleaning & EDA:** * Handled missing values in `TotalCharges`.
    * Analyzed correlations (e.g., Month-to-month contracts have higher churn rates).
2.  **Preprocessing:**
    * **One-Hot Encoding** for categorical variables.
    * **Feature Scaling** for numerical variables.
3.  **Handling Imbalance:**
    * Implemented **SMOTE (Synthetic Minority Over-sampling Technique)** to balance the training data (50:50 ratio).
4.  **Modeling & Tuning:**
    * **Baseline:** Random Forest Classifier (Default).
    * **Optimization:** Hyperparameter Tuning (`n_estimators`, `max_depth`) to prevent overfitting and improve generalization.

---

## 📊 Key Results & Evaluation
I compared three iterations of the model. The focus was on improving the **Recall score for Class 1 (Churn)**.

| Model Phase | Accuracy | Recall (Churn Detection) | Insight |
| :--- | :---: | :---: | :--- |
| **1. Baseline Model** | 79% | 46% | Missed more than half of the churners. |
| **2. With SMOTE** | 79% | 50% | Slightly better, but still underperforming. |
| **3. Final (Tuned + SMOTE)** | **80%** | **68%** | **Significant improvement (+22%).** |

### Visual Comparison

**Before Optimization (Baseline):**
> *Insert your first Confusion Matrix image here (The Blue One)*
![Baseline Confusion Matrix](path_to_your_blue_image.png)

**After Optimization (Tuned + SMOTE):**
> *Insert your final Confusion Matrix image here (The Orange One)*
![Final Confusion Matrix](path_to_your_orange_image.png)

---

## 🚀 Business Impact
By optimizing the model, we achieved a tangible impact on potential revenue retention:

* **Reduction in Missed Detections:** The Final Model successfully identified **253 churners**, compared to only 173 in the baseline.
* **Strategic Value:** This represents **80 additional high-risk customers** that the marketing team can now target with retention offers. Assuming an average customer lifetime value (CLV), saving these customers translates to significant revenue protection.

---

## 💻 Tech Stack
* **Language:** Python
* **Libraries:** * `Pandas`, `NumPy` (Data Manipulation)
    * `Matplotlib`, `Seaborn` (Visualization)
    * `Scikit-Learn` (Machine Learning)
    * `Imbalanced-Learn` (SMOTE)

## 📂 How to Run
1. Clone this repository.
2. Install the requirements:
   ```bash
   pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn
