"""
Inference pipeline for the Titanic Survival Prediction project.

Usage:
    python -m src.predict --model src/model.pkl --input test.csv --output submission.csv

This script loads a saved sklearn Pipeline and generates predictions
on new, unseen data.
"""

import argparse
import joblib
import pandas as pd

from .feature_engineering import engineer_features


def predict(model_path, input_path, output_path):
    """
    Load a trained pipeline and predict on new data.

    Parameters
    ----------
    model_path : str
        Path to the saved sklearn Pipeline (.pkl).
    input_path : str
        Path to the input CSV (must have the same columns as training data).
    output_path : str
        Path for the output submission CSV.
    """
    # load model
    pipeline = joblib.load(model_path)
    print(f'Loaded model from: {model_path}')

    # load and prepare data
    df = pd.read_csv(input_path)
    print(f'Input data shape: {df.shape}')

    # apply row-local feature engineering
    df = engineer_features(df)

    # define features (must match training features)
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare',
                'Embarked', 'Title', 'FamilySize', 'IsAlone', 'FarePerPerson']

    X = df[features]

    # generate predictions
    predictions = pipeline.predict(X)

    # create submission
    submission = pd.DataFrame({
        'PassengerId': df['PassengerId'],
        'Survived': predictions
    })

    submission.to_csv(output_path, index=False)
    print(f'Predictions saved to: {output_path}')
    print(f'Total predictions: {len(submission)}')
    print(f'Predicted survival rate: {predictions.mean()*100:.1f}%')

    return submission


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Titanic Survival Prediction')
    parser.add_argument('--model', required=True, help='Path to saved model (.pkl)')
    parser.add_argument('--input', required=True, help='Path to input CSV')
    parser.add_argument('--output', default='submission.csv', help='Output CSV path')

    args = parser.parse_args()
    predict(args.model, args.input, args.output)
