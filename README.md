# Metabolic Syndrome Classification Project
---
### Final Report
The **Final_Project_Report.pdf** is included in the repository. It offers a thorough description of the project, including the problem statement, literature review, data description, modeling techniques, and results. The report serves as a comprehensive guide to understanding the project's scope and contributions. Below is a brief summary of the project overview, but for a complete understanding, reading the full report is recommended. 

---
### Project Overview
This project focuses on the classification of **Metabolic Syndrome** using various machine learning models. The dataset used is from Kaggle and includes a variety of health-related features such as age, BMI, waist circumference, cholesterol levels, and more. The goal of this project is to develop accurate predictive models that can assist in early detection of Metabolic Syndrome, which can lead to serious health conditions such as diabetes, heart disease, and stroke. Several models were tested, including Logistic Regression, Random Forest, Neural Networks, and XGBoost.

---

### Dataset
The dataset used is titled `Metabolic Syndrome.csv`. It includes 2,401 observations with the following features:
- Age
- Sex
- Marital Status
- Income
- Race
- Waist Circumference
- BMI
- HDL (High-Density Lipoprotein Cholesterol)
- Triglycerides
- Blood Glucose
- Uric Acid Levels
- Urinary Albumin-Creatinine Ratio (UrAlbCr)

---

### Files Included
- **Metabolic Syndrome.csv**: The dataset used for analysis and model training.
- **README.md**: This documentation file.
- **dataCleaning.ipynb**: Contains the data cleaning process where missing values were handled, and feature scaling was performed.
- **kmeans_clustering_final_project.ipynb**: Analysis using K-Means clustering to explore any underlying clusters in the dataset.
- **logistic_regression_final_project.ipynb**: Logistic Regression model for classifying Metabolic Syndrome.
- **neural_network_final_project.ipynb**: Neural Network implementation for classification.
- **pca_final_project.ipynb**: Principal Component Analysis (PCA) performed to reduce dimensionality.
- **pca_logistic_regression_final_project.ipynb**: Logistic Regression applied after PCA to evaluate the effect of dimensionality reduction on model performance.
- **random_forest_final_project.ipynb**: Random Forest classifier implementation.
- **visualizations_final_project.ipynb**: Data visualizations showing the distribution of features and correlation between variables.
- **xgboost_final_project.ipynb**: XGBoost model implementation for classification.
- **Final_Project_Report.pdf**: The final report detailing the project’s results, methodology, and conclusions.
- **Requirements.txt** : All required dependencies to run the projcet.

---

### Methodology

#### 1. **Data Cleaning**
   The initial step involved cleaning the raw dataset to ensure its usability for model building. Our data cleaning steps included:
   - **Handling Missing Values**: Some features, such as Waist Circumference and BMI, contained missing values. Missing values were imputed using either the mean for continuous variables or the mode for categorical variables. For the neural network model, missing values were dropped altogether to avoid complications during training.
   - **Feature Scaling**: Many of the features had vastly different ranges, which could have negatively impacted model performance. To address this, **StandardScaler** from `sklearn` was used to normalize the data by scaling it to unit variance, ensuring that no single feature disproportionately influenced the model.

#### 2. **Exploratory Data Analysis (EDA)**
   EDA was performed to better understand the relationships between the features and the target variable (Metabolic Syndrome). Visualizations included:
   - **Distribution Analysis**: Histograms and box plots were generated to explore the distribution of continuous features like Age, BMI, and Waist Circumference. Categorical features such as Sex, Marital Status, and Race were visualized with bar plots.
   - **Correlation Matrix**: A heatmap was created to identify correlations between variables, helping to determine which features were most likely to influence the presence of Metabolic Syndrome.
   - **Class Balance**: The target variable (Metabolic Syndrome) was slightly imbalanced, with approximately 34.2% of the dataset labeled as having the syndrome.

#### 3. **Feature Engineering**
   To prepare the dataset for machine learning models, several feature engineering steps were taken:
   - **One-Hot Encoding**: Categorical features such as Sex, Marital Status, and Race were transformed using one-hot encoding. This converted these categorical variables into numerical form so they could be used in machine learning algorithms.
   - **Principal Component Analysis (PCA)**: PCA was applied to reduce the dimensionality of the dataset, retaining only the most important components. This step was used to see if dimensionality reduction would improve performance, particularly in conjunction with Logistic Regression.

#### 4. **Modeling**
   Several machine learning models were employed to classify Metabolic Syndrome:
   - **Logistic Regression**: A simple and interpretable model, Logistic Regression was used as a baseline for binary classification. After hyperparameter tuning, it achieved an accuracy of 82.5%.
   - **PCA + Logistic Regression**: After reducing the dataset to 3 principal components via PCA, Logistic Regression was applied to see if it could achieve similar performance with reduced data. The accuracy was 81.5%.
   - **XGBoost**: A powerful gradient boosting algorithm, XGBoost was employed due to its ability to handle structured data and provide high accuracy. It achieved an accuracy of 88.4% and an F1 score of 0.817.
   - **Random Forest**: Random Forest, an ensemble learning method, was also used. It performed slightly worse than XGBoost with an accuracy of 87%, but it was still a strong performer.
   - **Neural Network**: The Neural Network model was the most complex and achieved the highest accuracy at 91.4%. However, it was less interpretable compared to XGBoost.

#### 5. **Model Evaluation**
   Each model was evaluated based on:
   - **Cross-Validation Accuracy**: K-Fold Cross-Validation (with 5 folds) was used to evaluate the generalizability of each model.
   - **Accuracy on Test Set**: The accuracy of each model was calculated on the held-out test set to determine real-world performance.
   - **F1 Score**: This metric was used to balance precision and recall, ensuring that models did not just maximize accuracy but also reduced false positives and false negatives.
   - **Interpretability**: Given the high stakes in medical applications, the model's interpretability was critically assessed. This includes evaluating how easily healthcare professionals can understand the model's decision-making process and rationale. High interpretability is essential for ensuring trust, validating clinical decisions, and providing actionable insights for patient care.

   Results showed that the **Neural Network** had the best performance on the test set with an accuracy of 91.4%, followed closely by **XGBoost** at 88.4%. However, XGBoost was more interpretable and is potentially more useful in a healthcare setting where transparency is critical.

---

### How to Use
1. Clone the repository:
   ```bash
   git clone <repository-url>
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Open any of the Jupyter notebooks to explore the data cleaning, visualization, or model-building processes:
   ```bash
   jupyter notebook <notebook_name>.ipynb
   ```

---

### Future Work
Future enhancements could include:
- **Expanding the Dataset**: Including more diverse population groups to make the model more generalizable.
- **Improving Feature Selection**: Further refinement of features, such as focusing on variables most predictive of Metabolic Syndrome.
- **Additional Models**: Testing additional machine learning techniques or ensembles, such as Support Vector Machines or stacking models, to improve accuracy and interpretability.
