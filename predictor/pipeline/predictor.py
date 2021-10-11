import logging

from sklearn.base import BaseEstimator, ClassifierMixin

from predictor.utils import Singleton, load_pickle

logger = logging.getLogger(__name__)


class Predictor(BaseEstimator, ClassifierMixin, metaclass=Singleton):

    def __init__(self, model_path):
        self.model_path = model_path
        self.model = load_pickle(self.model_path)

    def fit(self, x, y):
        return self

    def predict(self, data):
        logger.info('making prediction...')
        return self.model.predict(data)
