# 🚢 Titanic Survival Prediction — Machine Learning Modeling

A machine learning project that predicts **passenger survival** on the Titanic using classification models. Built with Python, scikit-learn, and a robust preprocessing pipeline.

## 📋 Project Overview

| Item | Detail |
|------|--------|
| **Problem** | Binary classification — predict survival (1) or death (0) |
| **Dataset** | [Kaggle Titanic Dataset](https://www.kaggle.com/datasets/yasserh/titanic-dataset) |
| **Size** | 891 passengers × 12 features |
| **Target** | `Survived` (0 = No, 1 = Yes) |
| **Best Model** | Logistic Regression — **~85% accuracy**, **0.88 ROC-AUC** |

## 📊 Dataset Features

| Feature | Type | Description |
|---------|------|-------------|
| `Pclass` | Ordinal | Passenger class (1st, 2nd, 3rd) — proxy for socio-economic status |
| `Sex` | Categorical | Gender |
| `Age` | Continuous | Age in years (20% missing → imputed with **median**) |
| `SibSp` | Discrete | # of siblings/spouses aboard |
| `Parch` | Discrete | # of parents/children aboard |
| `Fare` | Continuous | Ticket fare |
| `Embarked` | Categorical | Port of embarkation (S/C/Q) (0.2% missing → imputed with **mode**) |
| `Cabin` | Categorical | Cabin number (77% missing → **dropped**) |

## 🔧 Project Pipeline

```
Raw Data → EDA → Data Cleaning → Feature Engineering → Pipeline (Preprocessing + Model) → Evaluation
```

### 1. Exploratory Data Analysis
- Missing data analysis with visualization
- Survival rate by Sex, Class, Age, Embarked
- Feature correlation heatmap
- Dataset imbalance analysis (38% survived, 62% died)

### 2. Feature Engineering
| Feature | Method | Rationale |
|---------|--------|-----------|
| `FamilySize` | `SibSp + Parch + 1` | Total family onboard |
| `IsAlone` | `1 if FamilySize == 1` | Solo travelers had different survival rates |
| `Title` | Extracted from `Name` | Social status indicator (Mr, Mrs, Miss, Master, Rare) |
| `FarePerPerson` | `Fare / FamilySize` | More accurate per-person cost |
| `FareGroup` | `pd.qcut(Fare, 4)` | Fare quartile bins (Low/Medium/High/VeryHigh) |
| `TicketGroupSize` | Count of shared tickets | Group travel indicator |

### 3. Preprocessing Pipeline
```python
Pipeline([
    ('preprocessor', ColumnTransformer([
        ('num', StandardScaler(), numeric_cols),        # Zero mean, unit variance
        ('cat', OneHotEncoder(drop='first'), cat_cols)   # Avoids multicollinearity
    ])),
    ('classifier', Model)
])
```
> Pipeline prevents **data leakage** — scaler is fit only on training data.

### 4. Models & Results

| Model | Test Accuracy | Precision | Recall | F1-Score | ROC-AUC | 10-Fold CV |
|-------|:---:|:---:|:---:|:---:|:---:|:---:|
| **Logistic Regression** | **0.85** | 0.84 | 0.75 | 0.80 | **0.88** | 0.82 ± 0.04 |
| Decision Tree | 0.84 | 0.81 | 0.76 | 0.78 | 0.85 | 0.80 ± 0.05 |
| Random Forest | 0.84 | 0.86 | 0.69 | 0.76 | 0.87 | 0.81 ± 0.04 |

All models tuned with **GridSearchCV** (5-fold cross-validation).

## 🔑 Key Findings

1. **Gender** is the strongest predictor — females had ~74% survival vs ~19% for males
2. **1st class** passengers survived at ~63%, 3rd class at ~24%
3. **Children** had significantly higher survival rates
4. **Family size 2–4** had best survival odds (vs solo or large families)
5. **Title** feature (extracted from Name) is highly predictive

## 🛠️ How to Run

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/titanic-survival-prediction.git
cd titanic-survival-prediction

# 2. Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn kagglehub

# 3. Run the notebook
jupyter notebook Titanic_Project_Improved.ipynb
```

## 📁 Repository Structure

```
├── Titanic_Project_Improved.ipynb   # Main notebook with full pipeline
├── Titanic-Dataset.csv              # Dataset
├── titanic_submission.csv           # Model predictions
├── README.md                        # This file
└── presentation.html                # Project presentation
```

## 📦 Dependencies

- Python 3.8+
- pandas, numpy
- scikit-learn
- matplotlib, seaborn
- kagglehub
