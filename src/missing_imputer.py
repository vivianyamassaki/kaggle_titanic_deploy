from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
import pickle
import pandas as pd


class MissingValuesImputer(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.imputer = self.load_imputer()

    def fit(self, df):
        self.imputer.fit(df)

        with open("models/imputer.pkl", "wb") as pickle_file:
            pickle.dump(self.imputer, pickle_file)

        return self

    def transform(self, df):
        columns = df.columns
        df = pd.DataFrame(self.imputer.transform(df), columns=columns)

        return df

    def load_imputer(self):
        try:
            with open("models/imputer.pkl", "rb") as pickle_file:
                return pickle.load(pickle_file)
        except FileNotFoundError:
            return SimpleImputer(strategy='median')
