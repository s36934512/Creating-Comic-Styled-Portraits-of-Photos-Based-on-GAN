import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models

from .basicnet import *


def stem(x_shape, filters):
    x_input = layers.Input(shape=x_shape)
    
    x_split = snconv(x_shape, filters=filters, strides=2, norm="layer")(x_input)
    
    x = snconv(x_split.shape[-3:], filters=filters // 2, kernel_size=1, norm="layer")(x_split)
    x = snconv(x.shape[-3:], filters=filters, strides=2, norm="layer")(x)

    y = layers.MaxPooling2D()(x_split)
    
    x = layers.Concatenate()([x, y])
    x = snconv(x.shape[-3:], filters=filters, norm="layer")(x)
    
    model = models.Model(inputs=[x_input], outputs=[x])
    return model


def context_embedding(x_shape, filters):
    x_input = layers.Input(shape=x_shape)
    
    x = layers.GlobalAveragePooling2D()(x_input)
    x = layers.LayerNormalization()(x)
    
    x = layers.Reshape((1, 1, filters))(x)
    
    x = snconv(x.shape[-3:], filters=filters, kernel_size=1, norm="layer")(x)
    
    x = layers.Add()([x, x_input])
    x = snconv(x.shape[-3:], filters=filters)(x)
    
    model = models.Model(inputs=[x_input], outputs=[x])
    return model


def bilateral_guided_aggregation(detail_shape, semantic_shape, filters):
    Detail = layers.Input(shape=detail_shape)
    Semantic = layers.Input(shape=semantic_shape)
    
    # detail branch
    detail_a = sndwconv(detail_shape, activation=None, norm="layer")(Detail)
    detail_a = snconv(detail_a.shape[-3:], filters=filters, kernel_size=1, activation=None)(detail_a)

    detail_b = snconv(detail_shape, filters=filters, strides=2, activation=None, norm="layer")(Detail)
    detail_b = layers.AveragePooling2D((3,3), strides=2, padding="same")(detail_b)
    
    # semantic branch
    semantic_a = sndwconv(semantic_shape, activation=None, norm="layer")(Semantic)
    semantic_a = snconv(semantic_a.shape[-3:], filters=filters, kernel_size=1, activation="sigmoid")(semantic_a)
    
    semantic_b = snconv(semantic_shape, filters=filters, activation=None, norm="layer")(Semantic)
    semantic_b = layers.UpSampling2D((4,4), interpolation="bilinear")(semantic_b)
    semantic_b = layers.Activation("sigmoid")(semantic_b)
    
    # combining
    detail = layers.Multiply()([detail_a, semantic_b])
    semantic = layers.Multiply()([semantic_a, detail_b])
    semantic = layers.UpSampling2D((4,4), interpolation="bilinear")(semantic)
    
    x = layers.Add()([detail, semantic])
    x = snconv(x.shape[-3:], filters=filters, activation=None, norm="layer")(x)
    
    model = models.Model(inputs=[Detail, Semantic], outputs=[x])
    return model


def content_encoder(x_shape, xs_shape):
    x_input = layers.Input(shape=x_shape)
    x_s = layers.Input(shape=xs_shape)

    x = snconv(x_shape, filters=64, strides=2, norm="layer")(x_input)
    x = sg_block(x.shape[-3:], filters=64)(x)

    x = snconv(x.shape[-3:], filters=64, strides=2, norm="layer")(x)
    x = sg_block(x.shape[-3:], filters=64)(x)

    x = snconv(x.shape[-3:], filters=128, strides=2, norm="layer")(x)
    x = sg_block(x.shape[-3:], filters=128)(x)

    x = bilateral_guided_aggregation(x.shape[-3:], xs_shape, filters=128)([x, x_s])
    
    model = models.Model(inputs=[x_input, x_s], outputs=[x])
    return model


def style_encoder(x_shape):
    x_input = layers.Input(shape=x_shape)

    x = stem(x_shape, filters=16)(x_input)

    x = sg_block(x.shape[-3:], filters=32, strides=2)(x)
    x = sg_block(x.shape[-3:], filters=64, strides=2)(x)
    x = sg_block(x.shape[-3:], filters=128, strides=2)(x)

    x = context_embedding(x.shape[-3:], filters=128)(x)

    model = models.Model(inputs=[x_input], outputs=[x])
    return model


def encoder(img_shape):
    x_input  = layers.Input(shape=img_shape)
    
    x_s = style_encoder(img_shape)(x_input)
    x_c = content_encoder(img_shape, x_s.shape[-3:])([x_input, x_s])

    model = models.Model(inputs=[x_input], outputs=[x_c, x_s])
    return model


def make_encoder(img_shape=(128, 128, 3)):
    x_input  = layers.Input(shape=img_shape)
    
    x_c, x_s = encoder(img_shape)(x_input)
    x = sc_head(x_c.shape[-3:], scale=8)(x_c)

    model = models.Model(inputs=[x_input], outputs=[x, x_c, x_s])
    return model
