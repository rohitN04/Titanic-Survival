# Titanic-Survival

## Problem Statement
The sinking of the Titanic is one of the most infamous shipwrecks in history. The objective of this project is to build a predictive machine learning model to determine what sorts of people were more likely to survive. By analyzing historical passenger data (such as age, gender, and ticket class), this project demonstrates a complete, end-to-end machine learning classification pipeline.

## Installation & Setup
To run this project locally, you will need Python installed on your machine. Follow these steps to set up the environment:

1. **Clone the repository:**
```bash
git clone https://github.com/rohitN04/Titanic-Survival.git
cd Titanic-Survival
```

2. **Install the required dependencies:
It is recommended to use a virtual environment. Install the packages using the provided requirements.txt file:
```bash
pip install -r requirements.txt
```

3. **Run the Notebook:
Open the Jupyter Notebook to view the code, visualizations, and model training process:
```bash
jupyter notebook Titanic_Project.ipynb
```

## Dataset Overview
The dataset used is the classic Kaggle Titanic dataset. 
* **Total Features:** 11 original columns (PassengerId, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked).
* **Target Variable:** `Survived` (Binary: 0 = Did not survive, 1 = Survived).

## Approach

### 1. Data Cleaning
* Dropped the `Cabin` column due to a high percentage of missing data.
* Filled missing `Embarked` points with the most frequent port (mode).
* Handled numerical missing values (like `Age`) programmatically within a Scikit-Learn preprocessor.

### 2. EDA & Insights
* Visualized survival rates using Seaborn and Matplotlib.
* Discovered strong correlations between survival and demographic factors (e.g., females and 1st-class passengers survived at drastically higher rates).
* Applied a Log-Transformation to the `Fare` column to reduce right-skewness and normalize the distribution.

### 3. Feature Engineering
Engineered several new features from the raw data to help the models find hidden patterns:
* **FamilySize:** Combined `SibSp` (siblings/spouses) and `Parch` (parents/children) plus 1 for the passenger.
* **IsAlone:** A binary flag indicating if a passenger was traveling entirely alone.
* **Title:** Extracted societal titles (Mr, Mrs, Miss, Master, Rare) from the `Name` string to better group passengers and infer social status.

### 4. Modeling
* Built an end-to-end **Scikit-Learn Pipeline** utilizing a `ColumnTransformer`. 
* Applied `StandardScaler` to numerical columns (`Age`, `Fare`, `FamilySize`) to ensure distance-based and linear models performed optimally without data leakage.
* Trained three classification algorithms: **Logistic Regression**, **Decision Tree**, and **Random Forest**.
* Utilized `GridSearchCV` with 5-fold cross-validation to systematically tune hyperparameters for each algorithm.

### 5. Evaluation
* Evaluated models using Accuracy, Precision, Recall, and F1-Score.
* Plotted Confusion Matrices for visual evaluation of False Positives and False Negatives.
* Generated an ROC Curve to evaluate the True Positive Rate vs. False Positive Rate.

### 6. Prediction
* Passed the held-out validation data through the final, tuned pipeline.
* Generated a final `titanic_submission.csv` file containing `PassengerId` and `Survived` predictions.

## Results & Findings
* **Winning Model:** The tuned **Logistic Regression** algorithm performed the best, achieving an impressive **85.15% accuracy** on the test set. 
* **Why it Won:** Logistic Regression benefited massively from the `StandardScaler` pipeline step and the log-transformation of the `Fare` column, allowing it to outperform the tree-based models.
* **Key Drivers:** Gender (`Sex_male`), Ticket Price (`Fare`), and Passenger Class (`Pclass`) were the heaviest contributors to the model's decision-making process. Social status and gender played a far more significant role in survival than the port of embarkation or family size.

## Tools & Libraries Used
* **Language:** Python
* **Data Manipulation:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn
* **Machine Learning:** Scikit-Learn (Pipelines, ColumnTransformer, StandardScaler, GridSearchCV, LogisticRegression, RandomForestClassifier, DecisionTreeClassifier)
* **Environment:** Jupyter Notebook / Google Colab
