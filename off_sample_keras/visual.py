from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image


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
        axes[fold_i, 1].set_ylim(bottom=0)


def plot_masks(mask, prob_masks):
    plot_images([mask] + prob_masks)


def plot_images(images):
    fig, ax = plt.subplots(1, len(images), figsize=(3 * len(images), 3))
    for i, img in enumerate(images):
        ax[i].imshow(img)
    plt.show()


def plot_top_mistakes(ds_path, ds_ions, y, y_prob, label, n=10):
    df = pd.DataFrame({'y': y, 'y_prob': y_prob})
    df['diff'] = np.abs(y - y_prob)
    y_pred = y_prob > 0.5
    top_m_df = df[(y == label) & ~(y == y_pred)].sort_values(by='diff', ascending=False).head(n)

    for i, r in top_m_df.iterrows():
        print(r)
        img_path = ds_path / ('off' if r.y == 1 else 'on') / f'{ds_ions[i]}.png'
        print(img_path)
        img = Image.open(img_path)
        plt.imshow(np.array(img)[:, :, 0])
        plt.show()