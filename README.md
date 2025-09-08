# **Predicting Customer Churn in Telecom**

##  **Project Overview**

This project aims to predict customer churn (whether a customer will leave the service) using telecom customer data. The dataset contains information about customer demographics, account details, usage patterns, and payment methods. By preprocessing the data and applying machine learning models such as Logistic Regression and Random Forest, we derive insights to help the telecom company identify at-risk customers and improve retention strategies.

---

##  **Dataset Description**

The dataset includes the following columns:

* `customerID`: Unique customer identifier
* `gender`, `SeniorCitizen`, `Partner`, `Dependents`: Demographic details
* `tenure`, `PhoneService`, `MonthlyCharges`, `TotalCharges`: Service usage and billing
* `Contract`, `PaymentMethod`, `InternetService`: Subscription and payment preferences
* `Churn`: Target variable indicating if the customer left the service

Some features are categorical while others are numeric. One-hot encoding was already applied to some columns such as `Contract`, `PaymentMethod`, and `InternetService`.

---

##  **Step 1 – Loading the Dataset**

We loaded the dataset using Pandas and checked its structure with `.info()` and `.head()` functions. This allowed us to understand data types, the presence of missing values, and initial patterns.

**Insight:**
The dataset contained a mix of numeric and categorical columns. Some features required further encoding before applying machine learning models.

---

##  **Step 2 – Inspecting Missing Values**

We checked for missing data using `.isnull().sum()` and confirmed that there were no missing values in any column. This ensured that we could proceed with preprocessing without imputation.

**Insight:**
The dataset was clean, which simplified further analysis and avoided introducing bias through imputation techniques.

---

##  **Step 3 – Encoding Categorical Features**

Although the dataset already had one-hot encoded columns for `Contract`, `PaymentMethod`, and `InternetService`, several other columns like `gender`, `Partner`, and `Dependents` were still in text format. Machine learning models require numeric inputs, so we applied encoding techniques:

* **Label Encoding** was used for binary columns such as `gender`, `Partner`, and `Dependents`.
* **One-hot Encoding** was applied to multi-class columns like `MultipleLines`, `OnlineSecurity`, `OnlineBackup`, etc.

This ensured that all categorical data was transformed into a format usable by the models.

**Insight:**
Encoding converted text labels into numerical data, allowing algorithms to process patterns and relationships in the dataset effectively.

---

##  **Step 4 – Feature Scaling**

We scaled numeric features (`tenure`, `MonthlyCharges`, `TotalCharges`) using `StandardScaler`. Scaling ensures that features with larger numeric ranges don’t dominate model training and allows algorithms like Logistic Regression to converge more efficiently.

**Insight:**
Feature scaling improved model performance by standardizing data ranges, ensuring fair comparison across features.

---

##  **Step 5 – Splitting the Dataset**

We split the dataset into training and testing subsets using a 70-30 split. This allowed us to train the model on one portion of the data and test its performance on unseen data to avoid overfitting.

**Insight:**
Data splitting ensures that models generalize well to new customers and prevents inflated accuracy scores from memorization.

---

##  **Step 6 – Model Training**

We trained two models to predict customer churn:

1. **Logistic Regression:**
   A simple and interpretable model suited for binary classification problems.

2. **Random Forest Classifier:**
   An ensemble model that combines multiple decision trees to improve prediction accuracy.

Both models were trained using the same training data and evaluated on the test set.

**Insight:**
Using multiple models allowed us to compare performance and choose the best approach for customer churn prediction.

---

##  **Step 7 – Model Evaluation**

We evaluated models using metrics like:

* **Accuracy**
* **Precision**
* **Recall**
* **F1-score**
* **ROC-AUC**

Confusion matrices were plotted to visualize true and false positives/negatives.

**Insight:**
The Random Forest model outperformed Logistic Regression, suggesting it better captured complex patterns in the dataset. The ROC-AUC score also validated the robustness of the model.

---

##  **Step 8 – Feature Importance**

We visualized feature importance from the Random Forest model to understand which factors contributed most to churn predictions. Key features like `tenure`, `Contract type`, and `MonthlyCharges` were found to influence customer churn significantly.

**Insight:**
This analysis helps identify actionable areas where telecom providers can target interventions, such as offering discounts to customers with shorter tenure or high monthly charges.

---

##  **Step 9 – Saving the Model**

We saved the trained Random Forest model using `joblib`, allowing it to be reused later for deployment or further analysis without retraining.

```python
import joblib
joblib.dump(rf, 'churn_model.pkl')
```

**Insight:**
Saving models ensures reproducibility and scalability, making it easier to integrate into customer retention tools.

---

## Conclusion

* Proper preprocessing (encoding, scaling) was critical for model performance.
* Random Forest provided better predictive power compared to Logistic Regression.
* The dataset’s feature structure offered insights into customer behavior patterns.
* The model can assist telecom companies in identifying customers at risk of churn and designing strategies to retain them.

---
