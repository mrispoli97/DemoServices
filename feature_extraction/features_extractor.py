from feature_extraction.features_extraction import PEFeatureExtractor
from pprint import pprint
from utility import utils
import os
import pandas as pd


class FeatureExtractor:

    def __init__(self):
        self.extractor = PEFeatureExtractor()

    def get_features(self, path, dst=None):
        bytes = utils.read_binary(path)

        features = [str(value) for value in self.extractor.feature_vector(bytes)]

        _, filename = os.path.split(path)
        name, _ = os.path.splitext(filename)
        features_df = {'feature_' + str(i): float(feature) for i, feature in enumerate(features)}
        dataframe = pd.DataFrame()
        dataframe = dataframe.append(features_df, ignore_index=True)
        if dst:
            dataframe.to_pickle(os.path.join(dst, name))
        return dataframe


if __name__ == '__main__':
    sample = r'C:\Users\mario\PycharmProjects\DemoServices\test\samples\adware\0cb9b7fdc027a8d6a2682bb7c0de4adce8dd8d9f89906919e3969bc453294f39'
    fe = FeatureExtractor()
    features = fe.get_features(path=sample, dst=r'C:\Users\mario\PycharmProjects\DemoServices\test\features')
