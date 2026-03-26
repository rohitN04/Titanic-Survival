"""
Feature engineering module for the Titanic Survival Prediction project.

Contains only ROW-LOCAL feature transformations that are safe to apply
before the train/test split (no data leakage risk).

Statistical features (imputation, scaling, encoding) are handled
inside sklearn Pipelines — see model_training.py.
"""

import pandas as pd
import numpy as np


def engineer_features(df):
    """
    Create row-local engineered features.

    Each feature depends only on its own row's values, so it is safe
    to compute before the train/test split.

    Features created:
        FamilySize    — SibSp + Parch + 1
        IsAlone       — 1 if FamilySize == 1, else 0
        Title         — extracted from Name, standardized
        FarePerPerson — Fare / FamilySize
        FareGroup     — binned fare categories (Low / Mid / High / VeryHigh)
        AgeGroup      — categorized age bands (Child / Teen / Adult / Senior)
        TicketFreq    — number of passengers sharing the same ticket

    Parameters
    ----------
    df : pd.DataFrame
        Raw Titanic dataframe (train or test).

    Returns
    -------
    pd.DataFrame
        Copy of the input with new columns added.
    """
    df = df.copy()

    # --- Family features ---
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

    # --- Title extraction from Name ---
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].replace(
        ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major',
         'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].replace('Mlle', 'Miss')
    df['Title'] = df['Title'].replace('Ms', 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')

    # --- Fare per person ---
    df['FarePerPerson'] = df['Fare'] / df['FamilySize']

    # --- Fare group (binned) ---
    df['FareGroup'] = pd.cut(
        df['Fare'],
        bins=[-1, 7.91, 14.454, 31.0, 600],
        labels=['Low', 'Mid', 'High', 'VeryHigh']
    )

    # --- Age group (categorized) ---
    df['AgeGroup'] = pd.cut(
        df['Age'],
        bins=[0, 12, 18, 60, 100],
        labels=['Child', 'Teen', 'Adult', 'Senior']
    )

    # --- Ticket frequency (how many share the same ticket) ---
    ticket_counts = df['Ticket'].value_counts()
    df['TicketFreq'] = df['Ticket'].map(ticket_counts)

    return df
