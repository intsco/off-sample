from keras.losses import binary_crossentropy
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPool2D
from keras import losses, Input, Model
from keras.callbacks import EarlyStopping
from keras.regularizers import l2, l1_l2
from keras.optimizers import SGD, Adam
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from collections import OrderedDict
import numpy as np
import copy
import sklearn
import keras
from keras.metrics import binary_accuracy
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import f1_score
from sklearn.linear_model.logistic import LogisticRegression
from scipy.stats import spearmanr

from off_sample_utils import resize_image


def create_cnn(input_shape=(64, 64, 1), opt=None,
               l1_a=0, l2_a=0.01,
               init_filters=8,
               dropout_p=0.5,
               dense_fn=(256, 256),
               act_f='relu',
               kernel_initializer='glorot_uniform',
               metrics=None):
    model = Sequential()

    strides = 1
    pool_size = (2, 2)

    # model.add(Conv2D(3, (1, 1), strides=strides, padding='same', input_shape=input_shape,
    #                  kernel_initializer=kernel_initializer, kernel_regularizer=l1_l2(l1_a, l2_a)))
    # model.add(BatchNormalization())
    # model.add(Activation(act_f))

    model.add(Conv2D(init_filters, (3, 3), strides=strides, padding='same', input_shape=input_shape,
                     kernel_initializer=kernel_initializer, kernel_regularizer=l1_l2(l1_a, l2_a)))
    model.add(BatchNormalization())
    model.add(Activation(act_f))
    model.add(MaxPool2D(pool_size=pool_size))

    model.add(Conv2D(init_filters, (3, 3), strides=strides, padding='same',
                     kernel_initializer=kernel_initializer, kernel_regularizer=l1_l2(l1_a, l2_a)))
    model.add(BatchNormalization())
    model.add(Activation(act_f))
    model.add(MaxPool2D(pool_size=pool_size))

    model.add(Conv2D(init_filters * 2, (3, 3), strides=strides, padding='same',
                     kernel_initializer=kernel_initializer, kernel_regularizer=l1_l2(l1_a, l2_a)))
    model.add(BatchNormalization())
    model.add(Activation(act_f))
    model.add(MaxPool2D(pool_size=pool_size))

    model.add(Conv2D(init_filters * 4, (3, 3), strides=strides, padding='same',
                     kernel_initializer=kernel_initializer, kernel_regularizer=l1_l2(l1_a, l2_a)))
    model.add(BatchNormalization())
    model.add(Activation(act_f))
    model.add(MaxPool2D(pool_size=pool_size))

    model.add(Conv2D(init_filters * 8, (3, 3), strides=strides, padding='same',
                     kernel_initializer=kernel_initializer, kernel_regularizer=l1_l2(l1_a, l2_a)))
    model.add(BatchNormalization())
    model.add(Activation(act_f))
    model.add(MaxPool2D(pool_size=pool_size))

    model.add(Conv2D(init_filters * 16, (2, 2), strides=strides, padding='valid',
                     kernel_initializer=kernel_initializer, kernel_regularizer=l1_l2(l1_a, l2_a)))
    model.add(BatchNormalization())
    model.add(Activation(act_f))

    model.add(Flatten())

    model.add(Dense(dense_fn[0], kernel_initializer=kernel_initializer, kernel_regularizer=l1_l2(l1_a, l2_a)))
    model.add(Activation(act_f))
    model.add(Dropout(dropout_p))

    model.add(Dense(dense_fn[1], kernel_initializer=kernel_initializer, kernel_regularizer=l1_l2(l1_a, l2_a)))
    model.add(Activation(act_f))
    model.add(Dropout(dropout_p))

    model.add(Dense(1, kernel_initializer=kernel_initializer, kernel_regularizer=l1_l2(l1_a, l2_a)))
    model.add(Activation('sigmoid'))

    if not metrics:
        metrics = [binary_accuracy]
    model.compile(loss=binary_crossentropy, optimizer=opt, metrics=metrics)
    return model


class OffSampleImageDataGenerator(ImageDataGenerator):

    def standardize(self, x):
        return super().standardize(x.copy())


class OffSampleKerasClassifier(KerasClassifier):

    def __init__(self, **sk_params):
        super().__init__(**sk_params)
        self.classes_ = np.arange(2)
        self.model = None
        self.data_gen = None
        self.mask_data_gen = None

    def check_params(self, params):
        pass

    def fit(self, x, y, **kwargs):
        self.set_params(**kwargs)

        print('create_model args: {}'.format(self.filter_sk_params(create_cnn)))
        self.model = create_cnn(**self.filter_sk_params(create_cnn))

        data_gen_args = dict(
            featurewise_center=True,
            featurewise_std_normalization=True,
            rotation_range=30,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            rescale=0.3)
        seed = 13
        self.data_gen = OffSampleImageDataGenerator(**data_gen_args)
        # self.mask_data_gen = OffSampleImageDataGenerator(**data_gen_args)
        self.data_gen.fit(x, augment=True, seed=seed)
        # self.mask_data_gen.fit(x, augment=True, seed=seed)

        print('fit_generator args: {}'.format(
            {k: v for k, v in self.filter_sk_params(self.model.fit_generator).items() if k != 'validation_data'}))
        fit_args = copy.deepcopy(self.filter_sk_params(self.model.fit_generator))
        fit_args.update(kwargs)

        print('flow args: {}'.format(self.filter_sk_params(ImageDataGenerator.flow)))
        flow_args = copy.deepcopy(self.filter_sk_params(ImageDataGenerator.flow))

        image_gen = self.data_gen.flow(x, y, seed=seed, **flow_args)
        # mask_gen = self.mask_data_gen.flow(masks, y, seed=seed, **flow_args)

        history = self.model.fit_generator(image_gen, steps_per_epoch=len(x) / flow_args['batch_size'],
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


def tta_predict(model, X_test):
    def flip_axis(x, axis):
        x = np.asarray(x).swapaxes(axis, 0)
        x = x[::-1, ...]
        x = x.swapaxes(0, axis)
        return x

    tta_list = []
    for i, img in enumerate(X_test):
        tta_list.extend([img, flip_axis(img, 0),
                         flip_axis(img, 1), flip_axis(flip_axis(img, 1), 0)])
    X_test_tta = np.stack(tta_list, axis=0)

    y_test_pred_cnn_tta = model.predict(X_test_tta)
    if y_test_pred_cnn_tta.ndim > 1:
        y_test_pred_cnn_tta = y_test_pred_cnn_tta[:,-1]  # handles case when second dim size = 1 or 2
    return y_test_pred_cnn_tta.reshape(-1, 4).mean(axis=1)


class KerasCNN(object):
    data_gen_args = dict(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=15,
        width_shift_range=0.15,
        height_shift_range=0.15,
        horizontal_flip=True,
        vertical_flip=True,
        shear_range=0.2,
        zoom_range=0.2)

    def __init__(self, image_shape, save_path='custom-cnn-weights.hdf5'):
        self.save_path = save_path
        self.data_gen = OffSampleImageDataGenerator(**self.data_gen_args)
        self.model = None
        self.args = dict(input_shape=(image_shape + (1,)),
                         opt=keras.optimizers.Adam(lr=5e-4),
                         l2_a=0.01,
                         init_filters=8,
                         dropout_p=0.5,
                         dense_fn=(256, 256),
                         act_f='relu',
                         kernel_initializer='glorot_uniform',
                         metrics=[keras.metrics.binary_accuracy])

    def fit(self, X_train, y_train, X_valid=None, y_valid=None,
            epochs=20, batch_size=32, seed=13):
        callbacks = []
        self.data_gen.fit(X_train)
        if X_valid is not None:
            validation_data = (self.data_gen.standardize(X_valid), y_valid)
            checkpointer = keras.callbacks.ModelCheckpoint(filepath=self.save_path,
                                                           monitor='val_binary_accuracy',
                                                           verbose=1, save_best_only=True)
            callbacks.append(checkpointer)
        else:
            validation_data = None

        self.model = create_cnn(**self.args)
        return self.model.fit_generator(self.data_gen.flow(X_train, y_train, batch_size=batch_size, seed=seed),
                                         epochs=epochs, validation_data=validation_data,
                                         steps_per_epoch=len(X_train) / batch_size,
                                         callbacks=callbacks)

    def predict(self, X_test, load_best=False):
        if load_best:
            self.model = create_cnn(**self.args)
            self.model.load_weights(self.save_path)
        X_test = self.data_gen.standardize(X_test)
        return tta_predict(self.model, X_test)


class KerasNN(object):
    @staticmethod
    def build_model(feature_n=None, l1_a=None, l2_a=None, lr=None):
        kernel_regularizer = l1_l2(l1_a, l2_a)
        model_in = Input(shape=(feature_n,))
        out = Dense(256, kernel_regularizer=kernel_regularizer)(model_in)
        out = BatchNormalization()(out)
        out = Activation('relu')(out)
        out = Dropout(0.5)(out)
        out = Dense(32, kernel_regularizer=kernel_regularizer)(out)
        out = BatchNormalization()(out)
        out = Activation('relu')(out)
        out = Dropout(0.5)(out)
        out = Dense(16, kernel_regularizer=kernel_regularizer)(out)
        out = BatchNormalization()(out)
        out = Activation('relu')(out)
        out = Dropout(0.5)(out)
        out = Dense(1, activation='sigmoid', kernel_regularizer=kernel_regularizer)(out)
        model = Model(model_in, out)
        model.compile(loss='binary_crossentropy',
                      optimizer=Adam(lr=lr),
                      metrics=['binary_accuracy'])
        return model

    def __init__(self, feature_n, save_path='custom-dense-nn-weights.hdf5'):
        self.save_path = save_path
        self.model = None
        self.args = dict(
            feature_n=feature_n,
            lr=0.001,
            l1_a=0,
            l2_a=0.01)

    def fit(self, X_train, y_train, X_valid=None, y_valid=None,
            epochs=5, batch_size=64):
        callbacks = []
        if X_valid is not None:
            validation_data = (X_valid, y_valid)
            checkpointer = keras.callbacks.ModelCheckpoint(filepath=self.save_path,
                                                           monitor='val_binary_accuracy',
                                                           verbose=1, save_best_only=True)
            callbacks.append(checkpointer)
        else:
            validation_data = None

        self.model = self.build_model(**self.args)
        history = self.model.fit(X_train, y_train,
                                 validation_data=validation_data,
                                 batch_size=batch_size, epochs=epochs, verbose=1,
                                 callbacks=callbacks)
        return history

    def predict(self, X_test, load_best=False):
        if load_best:
            self.model = self.build_model(**self.args)
            self.model.load_weights(self.save_path)
        # X_test = self.data_gen.standardize(X_test)
        return self.model.predict(X_test)[:, 0]


class SKLogisticRegression(object):

    def __init__(self, n_components=50):
        self.pca = sklearn.decomposition.TruncatedSVD(n_components=n_components)
        self.model = sklearn.linear_model.logistic.LogisticRegression(solver='lbfgs', max_iter=300, verbose=1)

    def fit(self, X_train, y_train):
        X_train_pca = self.pca.fit_transform(X_train)
        self.model.fit(X_train_pca, y_train)

    def predict(self, X_test):
        X_test_pca = self.pca.transform(X_test)
        y_pred = self.model.predict_proba(X_test_pca)[:, 1]
        y_pred[np.isnan(y_pred)] = y_pred[~np.isnan(y_pred)].mean()
        return y_pred


def pixel_corr_predict(y_p_pred, groups_p_test, X_test, groups_test,
                       masks, image_shape):
    y_pred = []
    for group in np.unique(groups_p_test):
        pred_mask = y_p_pred[groups_p_test == group].reshape(masks[group].shape)
        pred_mask = resize_image(pred_mask, image_shape)
        for img in X_test[groups_test == group]:
            sp_corr = spearmanr(img[:, :, 0].flatten(), pred_mask.flatten()).correlation
            sp_corr = (sp_corr + 1) / 2  # normalising
            y_pred.append(sp_corr)
    return np.asarray(y_pred)


class Blender(object):

    def __init__(self, cnn, nn, lr, image_shape):
        self.cnn, self.nn, self.lr = cnn, nn, lr
        self.image_shape = image_shape
        self.model = None
        self.standard_scaler = None

        self.X_blend_test = None

    def first_level_pred(self, X_test, groups_test,
                         X_p_test, groups_p_test, masks):
        y_test_pred_cnn = self.cnn.predict(X_test)

        y_p_test_pred = self.nn.predict(X_p_test)
        y_test_pred_nn = pixel_corr_predict(y_p_test_pred, groups_p_test, X_test, groups_test,
                                            masks, self.image_shape)

        y_p_test_pred = self.lr.predict(X_p_test)
        y_test_pred_lr = pixel_corr_predict(y_p_test_pred, groups_p_test, X_test, groups_test,
                                            masks, self.image_shape)
        return np.stack([y_test_pred_cnn,
                         y_test_pred_nn,
                         y_test_pred_lr], axis=1)

    def fit(self, X_valid, y_valid, groups_valid, X_p_valid, groups_p_valid, masks):
        X_blend_train = self.first_level_pred(X_valid, groups_valid,
                                              X_p_valid, groups_p_valid, masks)
        y_blend_train = y_valid

        self.standard_scaler = sklearn.preprocessing.StandardScaler()
        X_blend_train_scaled = self.standard_scaler.fit_transform(X_blend_train)

        self.model = sklearn.linear_model.logistic.LogisticRegressionCV(cv=10, solver='liblinear')
        self.model.fit(X_blend_train_scaled, y_blend_train)

    def predict(self, X_test, groups_test, X_p_test, groups_p_test, masks):
        self.X_blend_test = self.first_level_pred(X_test, groups_test,
                                                  X_p_test, groups_p_test, masks)
        X_blend_test_scaled = self.standard_scaler.transform(self.X_blend_test)
        y_blend_test_pred = self.model.predict_proba(X_blend_test_scaled)[:, 1]
        return y_blend_test_pred































