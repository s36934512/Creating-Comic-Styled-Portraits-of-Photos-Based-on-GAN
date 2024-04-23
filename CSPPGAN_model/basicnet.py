import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models


# Spectral Normalization Convolution
def snconv(x_shape, filters, kernel_size=3, strides=1, groups=1, activation="LeakyReLU", norm=None):
    
    x_input = layers.Input(shape=x_shape)
    x = tfa.layers.SpectralNormalization(layers.Conv2D(filters=filters, kernel_size=kernel_size,
                                                       strides=strides, padding="same", groups=groups))(x_input)
    
    if norm == "layer":
        x = layers.LayerNormalization()(x)
        
    if activation == "LeakyReLU":
        activation=layers.LeakyReLU()
        
    if activation != None:
        x = layers.Activation(activation)(x)
    
    model = models.Model(inputs=[x_input], outputs=[x])
    return model


# Spectral Normalization Depthwise Convolution
def sndwconv(x_shape, kernel_size=3, strides=1, activation="LeakyReLU", norm=None):
    filters = x_shape[-1]

    model = snconv(x_shape, filters=filters, kernel_size=kernel_size, 
                   strides=strides, groups=filters, activation=activation, norm=norm)
    return model


# Point-wise Layer Instance Normalization
def PoLIN(x_shape, use_bias=True):
    x_input = layers.Input(shape=x_shape)

    x_I = tfa.layers.InstanceNormalization(center=False, scale=False)(x_input)
    x_L = layers.LayerNormalization(center=False, scale=False)(x_input)
    x_c = layers.concatenate([x_I, x_L])
    x = tfa.layers.SpectralNormalization(layers.Conv2D(filters=x_shape[-1], kernel_size=1, padding="same", use_bias=use_bias))(x_c)

    model = models.Model(inputs=[x_input], outputs=[x])
    return model


# Adaptively Point-wise Layer Instance Normalization
def AdaPoLIN(x_shape):
    g_shape = (1, 1, x_shape[-1])

    x_input = layers.Input(shape=x_shape)
    gamma   = layers.Input(shape=g_shape)
    beta    = layers.Input(shape=g_shape)

    x = PoLIN(x_shape, use_bias=False)(x_input)
    x = layers.Multiply()([x, gamma])
    x = layers.add([x, beta])

    model = models.Model(inputs=[x_input, gamma, beta], outputs=[x])
    return model


# Sandglass block
def sg_block(x_shape, filters, e=6, strides=1, norm="layer"):
    d = x_shape[-1]
    
    x_input = layers.Input(shape=x_shape)
    
    x = sndwconv(x_shape, norm=norm)(x_input)
    x = snconv(x.shape[-3:], filters=d // e, kernel_size=1,  activation=None)(x)
    x = snconv(x.shape[-3:], filters=filters, kernel_size=1)(x)
    x = sndwconv(x.shape[-3:], strides=strides, norm=norm, activation=None)(x)

    if strides == 1 and d == filters:
        x = layers.add([x, x_input])
    
    model = models.Model(inputs=[x_input], outputs=[x])
    return model


# style convert head
def sc_head(x_shape, scale=1):
    x_input = layers.Input(shape=x_shape)

    x = layers.UpSampling2D(size=(scale, scale), interpolation="bilinear")(x_input)
    x = snconv(x.shape[-3:], filters=3, kernel_size=1, activation="sigmoid")(x)

    model = models.Model(inputs=[x_input], outputs=[x])
    return model
