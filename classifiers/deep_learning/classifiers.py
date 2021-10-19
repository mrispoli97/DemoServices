import numpy as np

from binary_to_image_converter.binary_to_image_converter import BIConverter
from tensorflow.keras import models, optimizers
import tensorflow as tf
from utility import utils
from tensorflow.python.keras import backend as K
import os
import config as cfg
from pprint import pprint
from tensorflow.keras import layers
from utility import utils

LABELS = ['adware', 'benign', 'crypto_miner', 'downloader', 'dropper', 'file_infector',
          'flooder', 'installer', 'packed', 'ransomware', 'spyware', 'worm']


class BaseModel:
    def __init__(self, config, checkpoint=None, verbose=False):
        self.config = utils.load_json(config)
        self.model = None
        self.metrics = []
        self._build_model()
        self.load(checkpoint)
        self.bic = BIConverter()

    # save function that saves the checkpoint in the path defined in the config file
    def save(self, checkpoint_path):
        if self.model is None:
            raise Exception("You have to build the model first.")

        print(f"Saving model... {checkpoint_path}")
        self.model.save_weights(checkpoint_path)

    # load latest checkpoint from the experiment path defined in the config file
    def load(self, checkpoint):
        if self.model is None:
            raise Exception("You have to build the model first.")

        if checkpoint:
            print("Loading model checkpoint {} ...\n".format(checkpoint))
            self.model.load_weights(checkpoint)

    def _get_model(self):
        raise NotImplementedError

    def _build_model(self, verbose=False):

        flatten = layers.Flatten()
        dense_layer = layers.Dense(
            self.config['model']['num_classes'],
            activation=self.config['model']['activation']
        )

        base_model = self._get_model()
        inp = base_model.input
        out = dense_layer(flatten(base_model.output))

        model = models.Model(inp, out)

        self.model = model

        if self.config['model']['optimizer']['name'].lower() == 'adam':
            optimizer = optimizers.Adam(
                learning_rate=self.config['model']['optimizer']['learning_rate'],
            )
        elif self.config['model']['optimizer']['name'].lower() == 'sgd':
            optimizer = optimizers.SGD(
                learning_rate=self.config['model']['optimizer']['learning_rate'],
                momentum=self.config['model']['optimizer']['momentum']
                if 'momentum' in self.config['model']['optimizer'] else 0,
            )
        else:
            optimizer = optimizers.Adam()

        if 'categorical_accuracy' in self.config['model']['metrics']:
            self.metrics.append(
                tf.keras.metrics.CategoricalAccuracy()
            )

        if 'auc' in self.config['model']['metrics']:
            self.metrics.append(
                tf.keras.metrics.AUC(name='auc')
            )

        if 'f1_score' in self.config['model']['metrics']:
            self.metrics.append(utils.f1_score)

        self.model.compile(
            loss=self.config['model']['loss'],
            optimizer=optimizer,
            metrics=self.metrics
        )
        if verbose:
            print(self.model.summary())

    def evaluate(self, test_x, test_y):
        return self.model.evaluate(test_x, test_y)

    def evaluate_generator(self, generator):
        return self.model.evaluate(generator)

    def predict_generator(self, test_generator, steps):
        return self.model.predict_generator(test_generator, steps)

    def predict(self, path):
        image = self.bic.convert(path)
        return self.predict_from_image(image)

    def predict_from_image(self, image):
        img = np.float32(image).reshape((224, 224, 1))
        array = np.asarray([img])
        predictions = self.model.predict(array)[0]
        label_index = np.argmax(predictions)
        label = LABELS[label_index]
        return label


class Resnet50(BaseModel):
    def __init__(self):
        config = os.path.join(cfg.ROOT, 'classifiers', 'deep_learning', 'configs', 'ResNet50', 'config.json')
        checkpoint = os.path.join(cfg.ROOT, 'models', 'ResNet50', 'weights.h5')
        super(Resnet50, self).__init__(config, checkpoint)

    def _get_model(self):
        resnet50 = tf.keras.applications.resnet50.ResNet50(
            include_top=False,
            weights=None,
            input_shape=(224, 224, 1),
            pooling=None,
        )
        return resnet50


class MobilenetV2(BaseModel):
    def __init__(self):
        config = os.path.join(cfg.ROOT, 'classifiers', 'deep_learning', 'configs', 'MobileNetV2', 'config.json')
        checkpoint = os.path.join(cfg.ROOT, 'models', 'MobileNetV2', 'weights.h5')
        super(MobilenetV2, self).__init__(config, checkpoint)

    def _get_model(self):
        mobilenetV2 = tf.keras.applications.MobileNetV2(
            include_top=False,
            weights=None,
            pooling=None,
            input_shape=(224, 224, 1)
        )

        return mobilenetV2


class VGG16(BaseModel):
    def __init__(self):
        config = os.path.join(cfg.ROOT, 'classifiers', 'deep_learning', 'configs', 'VGG16', 'config.json')
        checkpoint = os.path.join(cfg.ROOT, 'models', 'VGG16', 'weights.h5')
        super(VGG16, self).__init__(config, checkpoint)

    def _get_model(self):
        vgg16 = tf.keras.applications.VGG16(
            include_top=False,
            weights=None,
            pooling=None,
            input_shape=(224, 224, 1)
        )

        return vgg16


class Xception(BaseModel):
    def __init__(self):
        config = os.path.join(cfg.ROOT, 'classifiers', 'deep_learning', 'configs', 'Xception', 'config.json')
        checkpoint = os.path.join(cfg.ROOT, 'models', 'Xception', 'weights.h5')
        super(Xception, self).__init__(config, checkpoint)

    def _get_model(self):
        xception = tf.keras.applications.Xception(
            include_top=False,
            weights=None,
            pooling=None,
            input_shape=(224, 224, 1)
        )

        return xception


def test_model(paths, labels, model):
    errors = 0
    num_samples = len(paths)
    for file, label in zip(paths, labels):
        prediction = model.predict(file)
        if prediction != label:
            print(f"{label} has been classified as {prediction}")
            errors += 1
    print(f"PERFORMANCE: {num_samples - errors}/{num_samples}")


def test_MobileNetV2(paths, labels):
    print(f"TESTING MobileNetV2")
    model = MobilenetV2()
    test_model(paths, labels, model)


def test_ResNet50(paths, labels):
    print(f"TESTING ResNet50")
    model = Resnet50()
    test_model(paths, labels, model)


def test_VGG16(paths, labels):
    print(f"TESTING VGG16")
    model = VGG16()
    test_model(paths, labels, model)


def test_Xception(paths, labels):
    print(f"TESTING Xception")
    model = Xception()
    test_model(paths, labels, model)


def test_all_models():
    paths = []
    labels = []
    for family in os.listdir(os.path.join(cfg.ROOT, 'test', 'samples')):
        files = os.listdir(os.path.join(cfg.ROOT, 'test', 'samples', family))[:10]
        paths += [os.path.join(cfg.ROOT, 'test', 'samples', family, file) for file in files]
        labels += [family for file in files]
    test_MobileNetV2(paths, labels)
    test_ResNet50(paths, labels)
    test_Xception(paths, labels)
    test_VGG16(paths, labels)


def test_obfuscated():
    path = r"C:\Users\mario\PycharmProjects\DemoServices\test\obfuscated\0cb9b7fdc027a8d6a2682bb7c0de4adce8dd8d9f89906919e3969bc453294f39"
    test_VGG16([path], ['adware'])


def main():
    test_obfuscated()


if __name__ == '__main__':
    main()
