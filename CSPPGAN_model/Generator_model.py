import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models

from .basicnet import *


# Style Mapping Block
def style_mapping(x_shape):
    x_input = layers.Input(shape=x_shape)

    x = layers.GlobalAveragePooling2D()(x_input)
    x = tfa.layers.SpectralNormalization(layers.Dense(256, activation=layers.LeakyReLU()))(x)
    x = tfa.layers.SpectralNormalization(layers.Dense(256, activation=layers.LeakyReLU()))(x)
    x = tfa.layers.SpectralNormalization(layers.Dense(256, activation=layers.LeakyReLU()))(x)
    
    x = layers.Reshape((1, 1, x.shape[-1]))(x)
    gamma, beta = tf.split(x, num_or_size_splits=2, axis=-1)

    model = models.Model(inputs=[x_input], outputs=[gamma, beta])
    return model


# Fine-grained Style Transfer Block
def FST_block(x_shape, scale=2):
    filters = x_shape[-1]
    g_shape = (1, 1, filters)

    x_input = layers.Input(shape=x_shape)
    gamma   = layers.Input(shape=g_shape)
    beta    = layers.Input(shape=g_shape)

    x = layers.UpSampling2D(size=(scale, scale), interpolation="bilinear")(x_input)
    x = sg_block(x.shape[-3:], filters=filters)(x)
    x = AdaPoLIN(x.shape[-3:])([x, gamma, beta])
    x = sg_block(x.shape[-3:], filters=filters)(x)
    
    model = models.Model(inputs=[x_input, gamma, beta], outputs=[x])
    return model


# Backbone Block
def backbone(x_shape, s_shape):
    x_input = layers.Input(shape=x_shape)
    s_input = layers.Input(shape=s_shape)
    
    gamma, beta = style_mapping(s_shape)(s_input)
    
    x_1 = FST_block(x_shape)([x_input, gamma, beta])
    x_2 = FST_block(x_1.shape[-3:])([x_1, gamma, beta])
    x_3 = FST_block(x_2.shape[-3:])([x_2, gamma, beta])
    
    model = models.Model(inputs=[x_input, s_input], outputs=[x_1, x_2, x_3])
    return model


def make_generator(c_shape=(16, 16, 128), s_shape=(4, 4, 128)):
    x = layers.Input(shape=c_shape)
    y = layers.Input(shape=c_shape)
    z = layers.Input(shape=s_shape)

    bbb = backbone(c_shape, s_shape)
    
    # x
    x_1, x_2, x_3 = bbb([x, z])

    x_1 = sc_head(x_1.shape[-3:], scale=4)(x_1)
    x_2 = sc_head(x_2.shape[-3:], scale=2)(x_2)
    x_3 = sc_head(x_3.shape[-3:], scale=1)(x_3)
    
    # y
    y_1, y_2, y_3 = bbb([y, z])

    y_1 = sc_head(y_1.shape[-3:], scale=4)(y_1)
    y_2 = sc_head(y_2.shape[-3:], scale=2)(y_2)
    y_3 = sc_head(y_3.shape[-3:], scale=1)(y_3)

    model = models.Model(inputs=[x, y, z], outputs=[x_1, x_2, x_3, y_1, y_2, y_3])
    return model
