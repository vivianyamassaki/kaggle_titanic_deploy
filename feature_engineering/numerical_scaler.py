from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler

from predictor.utils import load_pickle, save_pickle


class NumericalFeaturesScaler(BaseEstimator, TransformerMixin):

    def __init__(self, numerical_features):
        self.features = numerical_features
        self.pickle_path = 'models/scaler.pkl'
        self.scaler = load_pickle(self.pickle_path, default=MinMaxScaler())

    def fit(self, df):
        self.scaler.fit(df[self.features])
        save_pickle(self.scaler, self.pickle_path)
        return self

    def transform(self, df):
        df[self.features] = self.scaler.transform(df[self.features])

        return df
