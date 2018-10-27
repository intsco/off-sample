from keras import backend as K
from keras.models import Model, Input
from keras.layers import Conv2D, MaxPooling2D, Dropout, UpSampling2D, Activation, BatchNormalization
from keras.layers.merge import concatenate
from keras.regularizers import l2
from keras.optimizers import Adam, RMSprop, SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, LambdaCallback, Callback
from keras.metrics import binary_accuracy
from keras.wrappers.scikit_learn import KerasClassifier


kernel_initializer = 'he_normal'
start_filters = 8
dropout_rate = 0.5
conv2d_kernel_size = (3, 3)
l2_a = 0.01


def conv_2_bn_dropout(filters, x):
    x = Conv2D(filters, conv2d_kernel_size, padding='same',
               kernel_initializer=kernel_initializer, kernel_regularizer=l2(l2_a))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Dropout(dropout_rate)(x)

    x = Conv2D(filters, conv2d_kernel_size, padding='same',
               kernel_initializer=kernel_initializer, kernel_regularizer=l2(l2_a))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    return x


def up_conv_concat(filters, prev_inputs, skip_inputs):
    x = UpSampling2D(size=(2, 2))(prev_inputs)
    x = Conv2D(filters, (2, 2), padding='same', kernel_initializer=kernel_initializer)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = concatenate([skip_inputs, x], axis=3)
    return x


def dice_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = y_true_f * y_pred_f
    score = (2. * K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1. - score


def bce_dice_loss(y_true, y_pred):
    return K.binary_crossentropy(y_true, K.sigmoid(y_pred)) + dice_loss(y_true, K.sigmoid(y_pred))


def bce_logdice_loss(y_true, y_pred):
    return K.binary_crossentropy(y_true, y_pred) - K.log(1. - dice_loss(y_true, y_pred))


def unet(pretrained_weights=None, input_size=(64, 64, 1), summary=False):
    inputs = Input(input_size)

    # zero encoder, dim reduction
    # conv0 = Conv2D(1, (1, 1), activation='relu', padding='same', kernel_initializer=kernel_initializer)(inputs)

    # first encoder 8=2^3
    conv1 = conv_2_bn_dropout(start_filters, inputs)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    # conv1 = Conv2D(start_neurons, (3, 3), activation='relu', padding='same', kernel_initializer=kernel_initializer)(inputs)
    # conv1 = conv2d_dropout(inputs, start_neurons)
    # conv1 = Conv2D(start_neurons, (3, 3), activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv1)
    # conv1 = conv2d_dropout(conv1, start_neurons)
    # pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    # second encoder 16=2^4
    # conv2 = Conv2D(2 * start_neurons, (3, 3), activation='relu', padding='same', kernel_initializer=kernel_initializer)(pool1)
    # conv2 = conv2d_dropout(pool1, 2*start_neurons)
    # conv2 = Conv2D(2 * start_neurons, (3, 3), activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv2)
    # conv2 = conv2d_dropout(conv2, 2*start_neurons)
    # pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv2 = conv_2_bn_dropout(2 * start_filters, pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    # third encoder 32=2^5
    # conv3 = Conv2D(4 * start_neurons, (3, 3), activation='relu', padding='same', kernel_initializer=kernel_initializer)(pool2)
    # conv3 = conv2d_dropout(pool2, 4*start_neurons)
    # conv3 = Conv2D(4 * start_neurons, (3, 3), activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv3)
    # conv3 = conv2d_dropout(conv3, 4*start_neurons)
    # pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv3 = conv_2_bn_dropout(4 * start_filters, pool2)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    # forth encoder 64=2^6
    # conv4 = Conv2D(8 * start_neurons, (3, 3), activation='relu', padding='same', kernel_initializer=kernel_initializer)(pool3)
    # conv4 = conv2d_dropout(pool3, 8*start_neurons)
    # conv4 = Conv2D(8 * start_neurons, (3, 3), activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv4)
    # conv4 = conv2d_dropout(conv4, 8*start_neurons)
    # pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    conv4 = conv_2_bn_dropout(8 * start_filters, pool3)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    # fifth bottom unit 128
    # conv5 = Conv2D(16 * start_neurons, (3, 3), activation='relu', padding='same', kernel_initializer=kernel_initializer)(pool4)
    # conv5 = conv2d_dropout(pool4, 16*start_neurons)
    # conv5 = Conv2D(16 * start_neurons, (3, 3), activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv5)
    # conv5 = conv2d_dropout(conv5, 16*start_neurons)
    conv5 = conv_2_bn_dropout(16 * start_filters, pool4)

    # sixth decoder
    # up6 = Conv2D(8 * start_neurons, (2, 2), activation='relu', padding='same', kernel_initializer=kernel_initializer)(
    #     UpSampling2D(size=(2, 2))(conv5))
    # concat6 = concatenate([conv4, up6], axis=3)
    concat6 = up_conv_concat(8 * start_filters, conv5, conv4)
    # conv6 = Conv2D(8 * start_neurons, (3, 3), activation='relu', padding='same', kernel_initializer=kernel_initializer)(concat6)
    # conv6 = Conv2D(8 * start_neurons, (3, 3), activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv6)
    conv6 = conv_2_bn_dropout(8 * start_filters, concat6)


    # seventh decoder
    # up7 = Conv2D(4 * start_filters, (2, 2), activation='relu', padding='same', kernel_initializer=kernel_initializer)(
    #     UpSampling2D(size=(2, 2))(conv6))
    # concat7 = concatenate([conv3, up7], axis=3)
    concat7 = up_conv_concat(4 * start_filters, conv6, conv3)
    # conv7 = Conv2D(4 * start_neurons, (3, 3), activation='relu', padding='same', kernel_initializer=kernel_initializer)(concat7)
    # conv7 = Conv2D(4 * start_neurons, (3, 3), activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv7)
    conv7 = conv_2_bn_dropout(4 * start_filters, concat7)

    # eighth decoder
    # up8 = Conv2D(2 * start_filters, (2, 2), activation='relu', padding='same', kernel_initializer=kernel_initializer)(
    #     UpSampling2D(size=(2, 2))(conv7))
    # concat8 = concatenate([conv2, up8], axis=3)
    concat8 = up_conv_concat(2 * start_filters, conv7, conv2)
    # conv8 = Conv2D(2 * start_neurons, (3, 3), activation='relu', padding='same', kernel_initializer=kernel_initializer)(concat8)
    # conv8 = Conv2D(2 * start_neurons, (3, 3), activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv8)
    conv8 = conv_2_bn_dropout(2 * start_filters, concat8)

    # ninth decoder
    # up9 = Conv2D(start_filters, (2, 2), activation='relu', padding='same', kernel_initializer=kernel_initializer)(
    #     UpSampling2D(size=(2, 2))(conv8))
    # concat9 = concatenate([conv1, up9], axis=3)
    concat9 = up_conv_concat(start_filters, conv8, conv1)
    # conv9 = Conv2D(start_neurons, (3, 3), activation='relu', padding='same', kernel_initializer=kernel_initializer)(concat9)
    # conv9 = Conv2D(start_neurons, (3, 3), activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv9)
    conv9 = conv_2_bn_dropout(start_filters, concat9)

    # tenth one-to-one conv layer
    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=conv10)
    model.compile(optimizer=Adam(lr=1e-4),
                  # loss='binary_crossentropy',
                  loss=bce_logdice_loss,
                  metrics=['binary_accuracy'])

    if summary:
        model.summary()

    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model
