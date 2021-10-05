from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler
import pickle


class NumericalFeaturesScaler(BaseEstimator, TransformerMixin):

    def __init__(self, numerical_features):
        self.features = numerical_features
        self.scaler = self.load_numerical_features_scaler()

    def fit(self, df):
        self.scaler.fit(df[self.features])

        with open("models/scaler.pkl", "wb") as pickle_file:
            pickle.dump(self.scaler, pickle_file)

        return self

    def transform(self, df):
        df[self.features] = self.scaler.transform(df[self.features])

        return df

    def load_numerical_features_scaler(self):
        try:
            with open("models/scaler.pkl", "rb") as pickle_file:
                return pickle.load(pickle_file)
        except FileNotFoundError:
            return MinMaxScaler()
