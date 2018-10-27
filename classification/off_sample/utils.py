from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd
from collections import OrderedDict
from sklearn.metrics import f1_score, average_precision_score, precision_score, recall_score, accuracy_score


# def OffSampleImageDataGenerator(ImageDataGenerator):
#
#     def __init__(**kwargs):
#         super().__init(**kwargs)
#
#     def standardize(self, x):
#         return ImageDataGenerator.standardize(self, x.copy())


class OffSampleImageDataGenerator(ImageDataGenerator):

    def standardize(self, x):
        return super().standardize(x.copy())


def plot_learning_graph(history_list, attempts):
    from matplotlib import pyplot as plt
    # n = len(history_list)
    n = 1
    f, axes = plt.subplots(n, 2, figsize=(12, n * 4))
    if len(axes.shape) == 1: axes = axes.reshape((1,) + axes.shape)
    t = f.suptitle('Train/valid accuracy', fontsize=12)
    f.subplots_adjust(top=0.95, wspace=0.4)
    points_n = len(history_list[0]['binary_accuracy'])

    epoch_list = list(range(1, points_n + 1))

    for fold_i in range(n):
        for attempt_i in range(attempts):
            i = fold_i * n + attempt_i
            axes[fold_i, 0].plot(epoch_list, history_list[i]['binary_accuracy'], color='blue')
            axes[fold_i, 0].plot(epoch_list, history_list[i]['val_binary_accuracy'], color='orange')
        axes[fold_i, 0].set_xticks(np.arange(0, points_n + 1, 5))
        axes[fold_i, 0].set_ylabel('Accuracy Value')
        axes[fold_i, 0].set_xlabel('Epoch')
        axes[fold_i, 0].set_title('Accuracy')
        axes[fold_i, 0].set_ylim(bottom=0.75)

    for fold_i in range(n):
        for attempt_i in range(attempts):
            i = fold_i * 5 + attempt_i
            axes[fold_i, 1].plot(epoch_list[1:], history_list[i]['loss'][1:], color='blue')
            axes[fold_i, 1].plot(epoch_list[1:], history_list[i]['val_loss'][1:], color='orange')
        axes[fold_i, 1].set_xticks(np.arange(0, points_n + 1, 5))
        axes[fold_i, 1].set_ylabel('Loss Value')
        axes[fold_i, 1].set_xlabel('Epoch')
        axes[fold_i, 1].set_title('Loss')


def calc_metrics(y_true, y_pred):
    y_pred_lab = np.around(y_pred)
    metrics = []
    classes = [(0, 'on'), (1, 'off')]
    for i, cl in classes:
        metrics.append(OrderedDict({
            'f1': f1_score(y_true[:,i], y_pred_lab[:,i]),
            # 'prec_recall_score': average_precision_score(y_true[:,i], y_pred[:,i]),
            'prec': precision_score(y_true[:,i], y_pred_lab[:,i]),
            'recall': recall_score(y_true[:,i], y_pred_lab[:,i]),
            'acc': accuracy_score(y_true[:,i], y_pred_lab[:,i]),
        }))
    return pd.DataFrame(metrics, index=['on', 'off'])


from matplotlib.cm import viridis

def gray_to_rgb(img):
    return viridis(img.squeeze())[:,:,:3]

def convert_to_rgb(X):
    if X.shape[-1] == 3:
        return X

    (img_n, rows, cols) = X.shape[:3]
    X_rgb = np.zeros((img_n, rows, cols, 3))
    for i in range(img_n):
        X_rgb[i] = gray_to_rgb(X[i])
    return X_rgb


def make_subset(u_groups, X, y, groups, to_rgb=False):
    mask = np.array([g in u_groups for g in groups])
    X_sub = X[mask]
    X_sub.setflags(write=False)
    y_sub = y[mask]
    y_sub.setflags(write=False)
    groups_sub = groups[mask]
    groups_sub.setflags(write=False)
    if to_rgb:
        X_sub = convert_to_rgb(X_sub)
    return X_sub, y_sub, groups_sub


def evaluate_model(X_test, y_test, model, data_gen=None):
    if data_gen:
        X_test = data_gen.standardize(X_test)
    if hasattr(model, 'predict_proba'):
        y_test_pred = model.predict_proba(X_test, verbose=1)
    else:
        y_test_pred = model.predict(X_test, verbose=1)
    test_metrics = calc_metrics(y_test, y_test_pred)
    return test_metrics
