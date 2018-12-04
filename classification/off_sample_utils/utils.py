from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd
from collections import OrderedDict
from sklearn.metrics import f1_score, average_precision_score, precision_score, recall_score, accuracy_score


def convert_to_2dim(y):
    y_ = y
    if y.ndim < 2:
        y_ = np.zeros(shape=(y.shape[0], 2))
        y_[:,0] = 1 - y
        y_[:,1] = y
    return y_


def calc_metrics(y_true, y_pred):
    y_true = convert_to_2dim(y_true)
    y_pred = convert_to_2dim(y_pred)

    y_pred_lab = (y_pred > 0.5).astype(int)
    metrics = []
    classes = [(0, 'on'), (1, 'off')]
    for i, cl in classes:
        metrics.append(OrderedDict({
            'f1': f1_score(y_true[:,i], y_pred_lab[:,i]),
            'prec': precision_score(y_true[:,i], y_pred_lab[:,i]),
            'recall': recall_score(y_true[:,i], y_pred_lab[:,i]),
            'acc': accuracy_score(y_true[:,i], y_pred_lab[:,i]),
        }))
    return pd.DataFrame(metrics, index=['on', 'off'])


def calc_regr_metrics(y_true, y_prob, ds_name=None, model=None):
    if y_true.shape[1] > 1:
        y_true = y_true[:,1]
    if y_prob.shape[1] > 1:
        y_prob = y_prob[:, 1]
    metrics = {
        'model': model_name,
        'expl_var': sklearn.metrics.explained_variance_score(y_true, y_prob),
        'mse': sklearn.metrics.mean_squared_error(y_true, y_prob),
        'auc': sklearn.metrics.roc_auc_score(y_true, y_prob),
    }
    if ds_name:
        metrics['group'] = ds_name
    if model is not None:
        if type(model) == str:
            model_name = model
        else:
            model_name = str(model.__class__).strip('<>\'').split('.')[-1]
        metrics['model'] = model_name
    return metrics


def check_tf_dev():
    import tensorflow as tf
    print(tf.__version__)
    from tensorflow.python.client import device_lib
    for d in device_lib.list_local_devices():
        print(d)