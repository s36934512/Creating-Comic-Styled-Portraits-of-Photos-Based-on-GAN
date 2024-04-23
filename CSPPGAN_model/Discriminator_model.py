import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models

from .basicnet import *

def discriminator(x_shape):
    x_input = layers.Input(shape=x_shape)

    x  = snconv(x_shape, filters=64, strides=2)(x_input)
    g1 = layers.GlobalAveragePooling2D()(x)
    
    x  = sg_block(x.shape[-3:], filters=64, strides=2)(x)
    g2 = layers.GlobalAveragePooling2D()(x)

    x = sg_block(x.shape[-3:], filters=64, strides=2)(x)
    x = sg_block(x.shape[-3:], filters=128, strides=2)(x)
    x = sg_block(x.shape[-3:], filters=128, strides=2)(x)
    x = snconv(x.shape[-3:], filters=1)(x)

    model = models.Model(inputs=[x_input], outputs=[x, g1, g2])
    return model

def discriminator_r(x_shape):
    x_input = layers.Input(shape=x_shape)

    x  = snconv(x_shape, filters=64, strides=2)(x_input)
    g1 = layers.GlobalAveragePooling2D()(x)
    
    x  = sg_block(x.shape[-3:], filters=64, strides=2)(x)
    g2 = layers.GlobalAveragePooling2D()(x)

    x = sg_block(x.shape[-3:], filters=64, strides=2)(x)
    x = sg_block(x.shape[-3:], filters=128, strides=2)(x)
    x = snconv(x.shape[-3:], filters=1)(x)

    model = models.Model(inputs=[x_input], outputs=[x, g1, g2])
    return model


def make_multi_scale_discriminator(x_shape):
    x_input = layers.Input(shape=x_shape)
    
    x = layers.GaussianNoise(0.1)(x_input)
    
    d1 = layers.AveragePooling2D(strides=2)(x)
    d2 = layers.AveragePooling2D(strides=2)(d1)

    x_1, x_11, x_12 = discriminator(x_shape)(x)
    x_2, x_21, x_22 = discriminator(d1.shape[-3:])(d1)
    x_3, x_31, x_32 = discriminator_r(d2.shape[-3:])(d2)

    model = models.Model(inputs=[x_input], outputs=[x_1, x_2, x_3, x_11, x_12, x_21, x_22, x_31, x_32])
    return model
