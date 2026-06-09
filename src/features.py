import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin

# ── 1. Column definitions ──────────────────────────────────────────────────

NUMERIC_FEATURES     = ['age', 'fnlwgt', 'hours-per-week']
SKEWED_FEATURES      = ['capital-gain', 'capital-loss']   # skew >1, 90%+ zeros
CATEGORICAL_FEATURES = ['workclass', 'marital-status', 'occupation',
                         'relationship', 'race', 'sex', 'native-country']
DROP_COLS            = ['education-num']   # redundant with education

EDUCATION_ORDER = [
    'Preschool','1st-4th','5th-6th','7th-8th','9th','10th','11th','12th',
    'HS-grad','Some-college','Assoc-voc','Assoc-acdm',
    'Bachelors','Masters','Prof-school','Doctorate'
]

# ── 2. Custom log1p transformer ────────────────────────────────────────────

class Log1pTransformer(BaseEstimator, TransformerMixin):
    """Apply log1p to highly skewed columns, then standard-scale."""
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        return np.log1p(X)

# ── 3. Sub-pipelines ───────────────────────────────────────────────────────

numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  StandardScaler())
])

skewed_pipeline = Pipeline([
    ('imputer',  SimpleImputer(strategy='median')),
    ('log1p',    Log1pTransformer()),
    ('scaler',   StandardScaler())
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

education_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OrdinalEncoder(categories=[EDUCATION_ORDER]))
])

# ── 4. Full preprocessor ───────────────────────────────────────────────────

preprocessor = ColumnTransformer([
    ('num',     numeric_pipeline,     NUMERIC_FEATURES),
    ('skewed',  skewed_pipeline,      SKEWED_FEATURES),
    ('cat',     categorical_pipeline, CATEGORICAL_FEATURES),
    ('edu',     education_pipeline,   ['education']),
], remainder='drop')   # drops education-num and any unlisted cols

# ── 5. Helper functions ────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered columns before the sklearn pipeline runs."""
    df = df.copy()

    # Strip leading/trailing whitespace from all string columns
    df = df.apply(lambda col: col.str.strip() if col.dtype == 'object' else col)
    
    # Drop duplicates
    df = df.drop_duplicates()

    # Drop redundant column
    df = df.drop(columns=DROP_COLS, errors='ignore')

    # New feature: capital difference
    df['capital_diff'] = df['capital-gain'] - df['capital-loss']

    # New feature: age bucket
    df['age_group'] = pd.cut(
        df['age'],
        bins=[0, 25, 35, 45, 55, 100],
        labels=['<25', '25-35', '35-45', '45-55', '55+']
    ).astype(str)

    return df

def get_X_y(df: pd.DataFrame):
    """Split features and target."""
    X = df.drop(columns=['income'])
    y = df['income']
    return X, y