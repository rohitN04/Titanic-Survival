"""
Model training module for the Titanic Survival Prediction project.

Builds sklearn Pipelines that handle all statistical preprocessing
(imputation, scaling, encoding) to prevent data leakage.
"""

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report
)

# default feature column groups
NUMERIC_COLS = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare',
                'FamilySize', 'IsAlone', 'FarePerPerson']
CATEGORICAL_COLS = ['Sex', 'Embarked', 'Title']


def build_pipeline(model, numeric_cols=None, categorical_cols=None):
    """
    Build an sklearn Pipeline with preprocessing and a classifier.

    Preprocessing (all fitted on training data only):
        - Numeric: SimpleImputer(median) → StandardScaler
        - Categorical: SimpleImputer(most_frequent) → OneHotEncoder

    Parameters
    ----------
    model : sklearn estimator
        The classifier to use as the last pipeline step.
    numeric_cols : list[str], optional
        Numeric feature column names.
    categorical_cols : list[str], optional
        Categorical feature column names.

    Returns
    -------
    sklearn.pipeline.Pipeline
    """
    if numeric_cols is None:
        numeric_cols = NUMERIC_COLS
    if categorical_cols is None:
        categorical_cols = CATEGORICAL_COLS

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ],
        remainder='drop'
    )

    return Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])


def train_and_evaluate(models_and_params, X_train, X_test, y_train, y_test,
                       numeric_cols=None, categorical_cols=None):
    """
    Train models with GridSearchCV and evaluate on the validation set.

    Parameters
    ----------
    models_and_params : dict
        Keys are model names, values are dicts with 'model' and 'params'.
    X_train, X_test : pd.DataFrame
    y_train, y_test : pd.Series
    numeric_cols, categorical_cols : list[str], optional

    Returns
    -------
    results : dict — per-model metrics
    best_models : dict — per-model best Pipeline
    """
    results = {}
    best_models = {}

    for name, config in models_and_params.items():
        pipeline = build_pipeline(config['model'], numeric_cols, categorical_cols)

        grid = GridSearchCV(
            pipeline, config['params'],
            cv=5, scoring='accuracy', n_jobs=-1, verbose=0
        )
        grid.fit(X_train, y_train)

        best_pipeline = grid.best_estimator_
        best_models[name] = best_pipeline

        y_pred = best_pipeline.predict(X_test)
        y_pred_proba = best_pipeline.predict_proba(X_test)[:, 1]

        train_acc = best_pipeline.score(X_train, y_train)
        test_acc = accuracy_score(y_test, y_pred)
        cv_scores = cross_val_score(best_pipeline, X_train, y_train,
                                    cv=10, scoring='accuracy')

        results[name] = {
            'Best Params': grid.best_params_,
            'Train Accuracy': train_acc,
            'Test Accuracy': test_acc,
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1-Score': f1_score(y_test, y_pred),
            'ROC-AUC': roc_auc_score(y_test, y_pred_proba),
            'CV Mean': cv_scores.mean(),
            'CV Std': cv_scores.std(),
            'y_pred_proba': y_pred_proba,
            'y_pred': y_pred
        }

        print(f'{name}: Test Acc={test_acc:.4f}, '
              f'F1={results[name]["F1-Score"]:.4f}, '
              f'AUC={results[name]["ROC-AUC"]:.4f}')

    return results, best_models
