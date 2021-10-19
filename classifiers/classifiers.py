from classifiers.deep_learning.classifiers import MobilenetV2, Resnet50, Xception, VGG16
from classifiers.machine_learning.classifiers import LightGBM, RandomForest, XGBoost


class Classifier:

    def __init__(self):
        pass
        # self._mobilenet = MobilenetV2()
        # self._resnet = Resnet50()
        # self._xception = Xception()
        # self._vgg = VGG16()
        # self._xgboost = None
        # self._lgbm = None
        # self._rf = None

    def predict(self, model, filepath):
        model = model.lower()
        if model == 'mobilenet':
            return MobilenetV2().predict(filepath)
        if model == 'resnet':
            return Resnet50().predict(filepath)
        if model == 'vgg':
            return VGG16().predict(filepath)
        if model == 'xception':
            return Xception().predict(filepath)
        if model in ['lightgbm', 'lgbm']:
            return LightGBM().predict(filepath)
        if model in ['random forest', 'rf']:
            return RandomForest().predict(filepath)
        if model == 'xgboost':
            return XGBoost().predict(filepath)
        raise ValueError(f"model {model} is invalid.")
