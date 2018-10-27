import copy
import numpy as np
from keras import Sequential
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import RandomizedSearchCV


class OffSampleImageDataGenerator(ImageDataGenerator):

    def standardize(self, x):
        return super(OffSampleImageDataGenerator, self).standardize(x.copy())


class OffSampleKerasClassifier(KerasClassifier):

    def __init__(self, create_model=None, **sk_params):
        super().__init__(**sk_params)
        self.create_model = create_model
        self.classes_ = np.arange(2)
        print(self, self.create_model)

    def check_params(self, params):
        pass

    def fit(self, x, y, **kwargs):
        self.set_params(**kwargs)
        print('create_model args: {}'.format(self.filter_sk_params(self.create_model)))
        self.model = self.create_model(**self.filter_sk_params(self.create_model))

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
        return f1_score(y[: ,1], y_pred_lab[: ,1]) # 0 - on, 1 - off

    def score(self, x, y, **kwargs):
        return self._target_class_f1_score(x, y, **kwargs)

    def predict_proba(self, x, **kwargs):
        x = self.data_gen.standardize(x)
        return KerasClassifier.predict_proba(self, x, **kwargs)


if __name__ == '__main__':
    params_dist = {
        'batch_size': [32],
        #     'epochs': [5, 10, 15, 20, 25, 30, 35, 40],
        'epochs': [25],
        'opt_alg': [Adam],
        'opt_lr': [1e-4],
        #     'opt_lr': [1e-3, 1e-4, 1e-5],
        #     'opt_decay': [0]
    }

    def get_comb_n(params_dist):
        comb_n = 1
        for p, vals in params_dist.items():
            comb_n *= len(vals)
        return comb_n


    def make_cv_gen(train_u_groups, valid_u_groups, groups_sample, n=3):
        for _ in range(n):
            train_mask = np.array([g in train_u_groups for g in groups_sample])
            valid_mask = np.array([g in valid_u_groups for g in groups_sample])
            yield train_mask, valid_mask


    cv_gen = make_cv_gen(train_u_groups, valid_u_groups, groups_sample, n=1)
    validator = RandomizedSearchCV(kclf, param_distributions=params_dist,
                                   n_iter=min(get_comb_n(params_dist), 10),
                                   n_jobs=1, cv=cv_gen, random_state=13)
    validator.fit(X_sample, y_sample);

