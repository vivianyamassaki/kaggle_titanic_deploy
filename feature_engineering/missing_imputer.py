from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
import pandas as pd

from predictor.utils import load_pickle, save_pickle


class MissingValuesImputer(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.pickle_path = 'models/imputer.pkl'
        self.imputer = load_pickle(self.pickle_path, default=SimpleImputer(strategy='median'))

    def fit(self, df):
        self.imputer.fit(df)
        save_pickle(self.imputer, self.pickle_path)

        return self

    def transform(self, df):
        columns = df.columns
        df = pd.DataFrame(self.imputer.transform(df), columns=columns)

        return df
