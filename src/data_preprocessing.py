"""
Data preprocessing module for the Titanic Survival Prediction project.

Handles data loading, exploration, and basic cleaning (row-level operations only).
Statistical transformations are handled inside sklearn Pipelines to prevent data leakage.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub


def load_data():
    """Load the Titanic train and test datasets from Kaggle."""
    path = kagglehub.dataset_download("shuofxz/titanic-machine-learning-from-disaster")
    train_df = pd.read_csv(path + '/train.csv')
    test_df = pd.read_csv(path + '/test.csv')
    return train_df, test_df


def explore_data_overview(df):
    """
    Print basic dataset information: .info(), .describe(), and .head().

    Parameters
    ----------
    df : pd.DataFrame
        Raw Titanic training dataframe.

    Returns
    -------
    pd.DataFrame
        Output of df.describe().
    """
    print("=== Dataset Info ===")
    print(df.info())
    print("\n=== Statistical Summary ===")
    summary = df.describe()
    print(summary)
    print("\n=== First Few Rows ===")
    print(df.head())
    print(f"\n=== Missing Values ===")
    print(df.isnull().sum())
    return summary


def analyze_missing_data(df):
    """Visualize missing values in the dataset."""
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({'Missing Count': missing, 'Missing %': missing_pct})
    missing_df = missing_df.sort_values('Missing %', ascending=False)

    cols_with_missing = missing_df[missing_df['Missing Count'] > 0].index
    if len(cols_with_missing) > 0:
        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.bar(cols_with_missing, missing_df.loc[cols_with_missing, 'Missing %'],
                      color=['#e74c3c' if pct > 50 else '#f39c12' if pct > 10 else '#3498db'
                             for pct in missing_df.loc[cols_with_missing, 'Missing %']])
        ax.set_ylabel('Missing %')
        ax.set_title('Missing Data by Feature', fontsize=14, fontweight='bold')
        for bar, pct in zip(bars, missing_df.loc[cols_with_missing, 'Missing %']):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{pct:.1f}%', ha='center', fontsize=10)
        plt.tight_layout()
        plt.savefig('missing_data_analysis.png', dpi=150, bbox_inches='tight')
        plt.show()

    print('=== Missing Values ===')
    return missing_df[missing_df['Missing Count'] > 0]


def plot_eda(df):
    """
    Create a 2×2 grid of exploratory data analysis plots.

    Plots generated:
        1. Survival count (bar chart)
        2. Survival by Sex
        3. Age distribution by survival status
        4. Fare distribution (box plot by class)

    Parameters
    ----------
    df : pd.DataFrame
        Raw Titanic training dataframe.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Exploratory Data Analysis', fontsize=16, fontweight='bold')

    # 1. Survival count
    sns.countplot(data=df, x='Survived', palette='pastel', ax=axes[0, 0])
    axes[0, 0].set_title('Survival Distribution')
    axes[0, 0].set_xticklabels(['Not Survived (0)', 'Survived (1)'])

    # 2. Survival by Sex
    sns.countplot(data=df, x='Sex', hue='Survived', palette='pastel', ax=axes[0, 1])
    axes[0, 1].set_title('Survival by Sex')

    # 3. Age distribution by survival
    axes[1, 0].hist([df[df['Survived'] == 0]['Age'].dropna(),
                     df[df['Survived'] == 1]['Age'].dropna()],
                    bins=30, label=['Not Survived', 'Survived'],
                    color=['#e74c3c', '#2ecc71'], alpha=0.7, stacked=False)
    axes[1, 0].set_title('Age Distribution by Survival')
    axes[1, 0].set_xlabel('Age')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].legend()

    # 4. Fare distribution by class
    sns.boxplot(data=df, x='Pclass', y='Fare', palette='pastel', ax=axes[1, 1])
    axes[1, 1].set_title('Fare Distribution by Class')
    axes[1, 1].set_ylim(0, 200)

    plt.tight_layout()
    plt.savefig('eda_plots.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_correlation_heatmap(df):
    """
    Plot a heatmap of correlations between numeric features.

    Parameters
    ----------
    df : pd.DataFrame
        Titanic dataframe (with engineered features preferred).
    """
    numeric_df = df.select_dtypes(include=[np.number])
    plt.figure(figsize=(10, 8))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', center=0,
                fmt='.2f', linewidths=0.5, square=True)
    plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png', dpi=150, bbox_inches='tight')
    plt.show()