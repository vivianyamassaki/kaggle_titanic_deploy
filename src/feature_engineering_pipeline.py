from abc import ABC

from sklearn.pipeline import Pipeline

from missing_imputer import MissingValuesImputer
from numerical_scaler import NumericalFeaturesScaler
from ohe_transformer import OneHotEncode


class FeatureEngineering(ABC):

    def get_pipeline(self):
        return Pipeline(
            [
                ("ohe", OneHotEncode()),
                ("missing_imputer", MissingValuesImputer()),
                ("numerical_scaler", NumericalFeaturesScaler(numerical_features=['Age', 'Fare', 'Pclass'])),

            ]
        )
