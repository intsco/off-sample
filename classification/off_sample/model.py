from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPool2D
from keras import losses
from keras.callbacks import EarlyStopping
from keras.regularizers import l2, l1_l2
from keras.optimizers import SGD, Adam
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from collections import OrderedDict
import math
import numpy as np
from keras.metrics import binary_accuracy
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import f1_score

image_shape = (64, 64, 1)


def create_model(opt_alg=Adam, opt_lr=1e-3, opt_decay=0, metrics=None):
    model = Sequential()

    alpha_l1, alpha_l2 = 0.0, 0.01
    strides = (1, 1)
    pool_size = (2, 2)
    init_filters = 8

    model.add(Conv2D(init_filters, (3, 3), strides=strides, padding='same', input_shape=image_shape,
                     kernel_regularizer=l1_l2(alpha_l1, alpha_l2)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=pool_size))

    model.add(Conv2D(init_filters * 2, (3, 3), strides=strides, padding='same',
                     kernel_regularizer=l1_l2(alpha_l1, alpha_l2)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=pool_size))

    model.add(Conv2D(init_filters * 4, (3, 3), strides=strides, padding='same',
                     kernel_regularizer=l1_l2(alpha_l1, alpha_l2)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=pool_size))

    model.add(Conv2D(64, (3, 3), strides=strides, padding='same',
                     kernel_regularizer=l1_l2(alpha_l1, alpha_l2)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=pool_size))

    model.add(Conv2D(128, (3, 3), strides=strides, padding='same',
                     kernel_regularizer=l1_l2(alpha_l1, alpha_l2)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=pool_size))

    model.add(Conv2D(256, (2, 2), strides=strides, padding='valid',
                     kernel_regularizer=l1_l2(alpha_l1, alpha_l2)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Flatten())

    model.add(Dense(256, kernel_regularizer=l1_l2(alpha_l1, alpha_l2)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(256, kernel_regularizer=l1_l2(alpha_l1, alpha_l2)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(2, kernel_regularizer=l1_l2(alpha_l1, alpha_l2)))
    model.add(Activation('softmax'))

    if not metrics:
        metrics = [binary_accuracy]
    #                   optimizer=SGD(lr=0.005, momentum=0.9, decay=0, nesterov=False),
    #                   optimizer=Adam(lr=1e-4, decay=1e-5),
    model.compile(loss=losses.binary_crossentropy,
                  optimizer=opt_alg(lr=opt_lr, decay=opt_decay),
                  metrics=metrics)
    return model


import copy
from keras.preprocessing.image import ImageDataGenerator


class OffSampleImageDataGenerator(ImageDataGenerator):

    def standardize(self, x):
        return super(OffSampleImageDataGenerator, self).standardize(x.copy())


class OffSampleKerasClassifier(KerasClassifier):

    def __init__(self, **sk_params):
        super().__init__(**sk_params)
        self.classes_ = np.arange(2)

    def check_params(self, params):
        pass

    def fit(self, x, y, **kwargs):
        self.set_params(**kwargs)

        print('create_model args: {}'.format(self.filter_sk_params(create_model)))
        self.model = create_model(**self.filter_sk_params(create_model))

        self.data_gen = OffSampleImageDataGenerator(
            featurewise_center=True,
            featurewise_std_normalization=True,
            rotation_range=30,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            rescale=0.3)
        self.data_gen.fit(x)

        print('fit_generator args: {}'.format(self.filter_sk_params(Sequential.fit_generator)))
        fit_args = copy.deepcopy(self.filter_sk_params(Sequential.fit_generator))
        fit_args.update(kwargs)
        print('flow args: {}'.format(self.filter_sk_params(ImageDataGenerator.flow)))
        flow_args = copy.deepcopy(self.filter_sk_params(ImageDataGenerator.flow))

        history = self.model.fit_generator(self.data_gen.flow(x, y, **flow_args),
                                           **fit_args)
        return history

    def _target_class_f1_score(self, x, y, **kwargs):
        x = self.data_gen.standardize(x)
        y_pred = self.model.predict(x)
        y_pred_lab = np.around(y_pred)
        return f1_score(y[:, 1], y_pred_lab[:, 1])  # 0 - on, 1 - off

    def score(self, x, y, **kwargs):
        return self._target_class_f1_score(x, y, **kwargs)

    def predict_proba(self, x, **kwargs):
        x = self.data_gen.standardize(x)
        return KerasClassifier.predict_proba(self, x, **kwargs)


if __name__ == '__main__':
    print(create_model().summary())
