from obfuscation.obfuscator.obfuscator import Obfuscator
from classifiers.classifiers import Classifier
import config as cfg
import os


class Action:

    def __init__(self):
        self._obfuscator = Obfuscator()
        self._classifier = Classifier()

    def obfuscate(self, config, filepath, dst):
        return self._obfuscator.obfuscate(config=config, filepath=filepath, dst=dst)

    def classify(self, model, filepath):
        return self._classifier.predict(model=model, filepath=filepath)


def test():
    sample = r"C:\Users\mario\PycharmProjects\DemoServices\test\samples\adware\0cb9b7fdc027a8d6a2682bb7c0de4adce8dd8d9f89906919e3969bc453294f39"
    dst = os.path.join(cfg.ROOT, 'test', 'tmp')
    action = Action()
    for model in ['mobilenet', 'resnet', 'vgg', 'xception', 'lgbm', 'rf', 'xgboost']:
        print(f"\nMODEL: {model}")
        before = action.classify(model=model, filepath=sample)
        print(f"Before: {before}")
        config = {
            "obfuscation": "junk",
            "params": {
                "severity": 0.1
            }
        }
        obfuscated_sample = action.obfuscate(config=config, filepath=sample, dst=dst)
        after = action.classify(model=model, filepath=obfuscated_sample)
        print(f"After: {after}")


def main():
    test()


if __name__ == '__main__':
    main()
