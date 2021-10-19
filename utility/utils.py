import pickle as pkl
from tensorflow.python.keras import backend as K
import json


def load_pickle(filepath):
    with open(filepath, 'rb') as handle:
        data = pkl.load(handle)
    return data


def read_binary(filepath):
    with open(filepath, 'rb') as f:
        bytes = f.read()
    return bytes


def f1_score(y_true, y_pred):  # taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1_val


def load_json(filepath):
    print(filepath)
    with open(filepath, encoding='utf-8', mode='r') as f:
        data = json.load(f)
    return data


def get_percentage(current_step, num_steps):
    return current_step / num_steps * 100 if num_steps > 0 else 100
