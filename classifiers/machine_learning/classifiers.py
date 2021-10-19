import numpy as np

from utility import utils
from pprint import pprint
import os
import config as cfg
from feature_extraction.features_extractor import FeatureExtractor


class MachineLearning:

    def __init__(self, path, verbose=False):
        self.load(path, verbose=verbose)
        self.fe = FeatureExtractor()

    def load(self, path, verbose=False):
        print(f"Loading... {path}")
        self._automl = utils.load_pickle(path)

    def predict(self, path):
        features = self.fe.get_features(path=path)
        return self.predict_from_features(features)[0]

    def predict_from_features(self, features):
        y_pred = self._automl.predict(features)
        return y_pred


class LightGBM(MachineLearning):

    def __init__(self, verbose=False):
        path = os.path.join(cfg.ROOT, 'models', 'LightGBM', 'model.pkl')
        super().__init__(path, verbose)


class XGBoost(MachineLearning):
    def __init__(self, verbose=False):
        path = os.path.join(cfg.ROOT, 'models', 'XGBoost', 'model.pkl')
        super().__init__(path, verbose)


class RandomForest(MachineLearning):
    def __init__(self, verbose=False):
        path = os.path.join(cfg.ROOT, 'models', 'RandomForest', 'model.pkl')
        super().__init__(path, verbose)


def test_lgbm(paths):
    lgbm = LightGBM()
    for path in paths:
        predictions = lgbm.predict(path)
        pprint(predictions)


def test_xgboost(paths):
    xgboost = XGBoost()
    for path in paths:
        predictions = xgboost.predict(path)
        pprint(predictions)


def test_rf(paths):
    rf = RandomForest()
    for path in paths:
        predictions = rf.predict(path)
        pprint(predictions)


def main():
    files = os.listdir(os.path.join(cfg.ROOT, 'test', 'samples', 'dropper'))
    paths = [os.path.join(cfg.ROOT, 'test', 'samples', 'dropper', file) for file in files]
    test_rf(paths)


if __name__ == '__main__':
    main()
