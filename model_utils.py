import numpy as np
import sys
import os

from keras.models import Model
from keras import layers
from keras.layers import Lambda, Input, Dense, Flatten, Conv2D, Reshape, UpSampling2D, GlobalAveragePooling2D
from keras.layers import BatchNormalization, Concatenate, LeakyReLU
from keras.losses import sparse_categorical_crossentropy, categorical_crossentropy
from keras import backend as K
from keras.optimizers import Adam, RMSprop
import tensorflow as tf
import importlib

K.set_epsilon(1E-4)


def init(definitions_package):
    #  import * from <definitions_package>
    package = importlib.import_module(definitions_package)
    for object_name in package.__dict__:
        globals()[object_name] = getattr(package, object_name)


def convolution_block(input_layer, kernel_definitions, batch_normalization=True):
    """
    generate a simple block of multiple convolutional layers
    :param input_layer: input_layer
    :param kernel_definitions: list of pairs (n_filters, kernel_size)
    :param batch_normalization: True if BatchNormalization is to eb applied
    :return: return layer
    """
    if batch_normalization:
        current_layer = BatchNormalization(axis=-1)(input_layer)
    else:
        current_layer = input_layer
    for index, kernel_def in enumerate(kernel_definitions):
        n_filters, kernel_size = kernel_def
        current_layer = Conv2D(n_filters, kernel_size, dilation_rate=(1, 1), strides=(1, 1), padding='same')(
            current_layer)
        current_layer = LeakyReLU(alpha=ALPHA)(current_layer)

    return current_layer


def downsample(input_layer, stride, kernel_def, batch_normalization=True):
    """
    apply a single convolutional layer, with stride larger than 1
    """
    n_filters, kernel_size = kernel_def
    shape = K.int_shape(input_layer)
    name = "down_%d_to_%d" % (shape[1], shape[1] * stride)
    if batch_normalization:
        input_layer = BatchNormalization(axis=-1)(input_layer)
    current_layer = Conv2D(n_filters, kernel_size, name=name, strides=(stride, stride), padding='same')(input_layer)
    current_layer = LeakyReLU(alpha=ALPHA)(current_layer)
    return current_layer


def upsample(input_layer, factor, kernel_def, batch_normalization=True):
    """
    apply an upsampling layer and afterwards a single convolutional layer
    """
    n_filters, kernel_size = kernel_def
    shape = K.int_shape(input_layer)
    name = "up_%d_to_%d" % (shape[1], shape[1] * factor)
    current_layer = UpSampling2D(size=(factor, factor), name=name)(input_layer)
    if batch_normalization:
        current_layer = BatchNormalization(axis=-1)(current_layer)
    current_layer = Conv2D(n_filters, kernel_size, dilation_rate=(1, 1), strides=(1, 1), padding='same')(current_layer)
    current_layer = LeakyReLU(alpha=ALPHA, name=name + "_activation")(current_layer)
    return current_layer


def make_generator():
    """
    constructs the generator model, following a U-net architecture
    there are two different flavours:
        * one pre_generator which produces two distinct a-b layers, hot encoding the two discreticed color channels
        * one generator, producing a L-a-b image (original L channel + two generated a-b- channels)
    """
    encode_0 = Input(IMAGE_SHAPE_1, name="input_layer")

    # encoder with downsampling
    encode_1 = convolution_block(encode_0, [(32, 11)], True)
    encode_2 = downsample(encode_1, 2, (32, 5), False)
    encode_4 = downsample(encode_2, 2, (64, 5), False)
    encode_8 = downsample(encode_4, 2, (128, 5), True)
    encode_16 = downsample(encode_8, 2, (256, 5), False)
    encode_32 = downsample(encode_16, 2, (256, 5), False)
    encode_64 = downsample(encode_32, 2, (512, 5), False)

    # decoder with upsampling and skip connections
    decode_64 = convolution_block(encode_64, [(256, 5)])
    decode_32 = upsample(decode_64, 2, (256, 5), False)

    decode_32m = Concatenate(axis=-1)([encode_32, decode_32])
    decode_16 = upsample(decode_32m, 2, (256, 5), True)

    decode_16m = Concatenate(axis=-1)([encode_16, decode_16])
    decode_8 = upsample(decode_16m, 2, (256, 5), False)

    decode_8m = Concatenate(axis=-1)([encode_8, decode_8])
    decode_4 = upsample(decode_8m, 2, (128, 5), False)

    decode_4m = Concatenate(axis=-1)([encode_4, decode_4])
    decode_2 = upsample(decode_4m, 2, (64, 5), True)

    decode_2m = Concatenate(axis=-1)([encode_2, decode_2])
    decode_1 = upsample(decode_2m, 2, (64, 5), False)

    # generate outputs
    discrete_a = Conv2D(BINS, 5, dilation_rate=(1, 1), padding='same')(decode_1)
    discrete_a = layers.Softmax(axis=-1, name="discrete_a")(discrete_a)
    discrete_b = Conv2D(BINS, 5, dilation_rate=(1, 1), padding='same')(decode_1)
    discrete_b = layers.Softmax(axis=-1, name="discrete_b")(discrete_b)
    discrete_model = Model(encode_0, [discrete_a, discrete_b])

    gauss_a = Conv2D(1, 5, padding='same', activation='tanh', name='gauss_a')(decode_1)
    gauss_b = Conv2D(1, 5, padding='same', activation='tanh', name='gauss_b')(decode_1)
    output_image = Concatenate(axis=-1)([encode_0, gauss_a, gauss_b])
    continuous_model = Model(encode_0, output_image)

    return discrete_model, continuous_model


def make_critic_64():
    """
    main critic: uses several downsampling blocks and eventually several dense layers to aggregate the result
    there are also several shortcuts directly connecting a layer into the dense block
    """
    encode_0 = Input(IMAGE_SHAPE_3, name="input_layer")
    encode_0b = Lambda(lambda x: x[:, RIM:-RIM, RIM:-RIM, :], output_shape=REDUCED_SHAPE_3)(encode_0)
    encode_1 = convolution_block(encode_0b, [(64, 7), (64, 7)], False)

    # encoder with downsampling
    encode_2 = downsample(encode_1, 2, (128, 5), False)
    encode_4 = downsample(encode_2, 2, (128, 5), False)
    encode_8 = downsample(encode_4, 2, (256, 5), False)
    encode_16 = downsample(encode_8, 2, (512, 3), False)
    encode_32 = downsample(encode_16, 2, (512, 3), False)
    encode_64 = downsample(encode_32, 2, (512, 3), False)

    skip1 = GlobalAveragePooling2D(name='skip1')(encode_1)
    skip1 = Dense(4)(skip1)
    skip4 = GlobalAveragePooling2D(name='skip4')(encode_4)
    skip4 = Dense(4)(skip4)
    skip32 = GlobalAveragePooling2D(name='skip32')(encode_32)
    skip32 = Dense(4)(skip32)

    flat_0 = Flatten()(encode_64)
    flat_1 = Dense(1024)(flat_0)
    flat_1 = LeakyReLU(alpha=ALPHA)(flat_1)
    flat_2 = Dense(256)(flat_1)

    flat_2x = Concatenate(axis=-1)([flat_2, skip1, skip4, skip32])
    flat_2x = LeakyReLU(alpha=ALPHA)(flat_2x)

    flat_3 = Dense(64)(flat_2x)
    flat_3 = LeakyReLU(alpha=ALPHA)(flat_3)
    flat_4 = Dense(16)(flat_3)
    flat_4 = LeakyReLU(alpha=ALPHA)(flat_4)
    flat_5 = Dense(1, name='score_output')(flat_4)

    return Model(encode_0, flat_5)


class ModelContainer:
    """
    wrapper for a model, that switches layers between trainable/not-trainable
    """
    def __init__(self, model):
        self.model = model

    def apply(self, input_layer, is_trainable: bool, freeze_lower = 0):
        for layer in self.model.layers:
            layer.trainable = is_trainable

        for ind, layer in enumerate(self.model.layers):
            if ind < freeze_lower:
                layer.trainable = False
        return self.model(input_layer)


def interpolate_layers(args):
    """ takes two layers of same shape as input and returns a random intermediate tensor
    """
    layer1, layer2 = args
    weights = K.random_uniform((BATCH_SIZE, 1, 1, 1))
    return (weights * layer1) + ((1 - weights) * layer2)


def calculate_gradient_penalty(args):
    """
    calculates the gradients of output_layer (at the given positions) over the variables stated in input_layer
    input_layer : tensor of format (batch_dim, None)
    output_layer : tensor of format (batch_dim, data_dim1. data_dim2,....)
    returns : should be a tensor of (batch_dim, None)
    """
    #easier to describe in tensorflow
    output_layer, input_layer = args
    ndims = len(input_layer.get_shape())
    data_dimensions = list(range(1, ndims))  # all axis besides batch_dim must be summed over

    gradients = tf.gradients(output_layer, input_layer)[0]  # returns a list with length 1
    gradients_norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=data_dimensions))
    gp = tf.square(gradients_norm - 1.0) * LAMBDA_GP

    return gp


def weighted_mean_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)


def mean_loss(y_true, y_pred):
    return K.mean(y_pred)


def generate_and_compile_models():
    """
    create all main models
    :return: a list of various models:
    *       pre_generator : only the generator, producing one-hot encoded a-b channels
            generator : yielding L-a-b image
            generator_trainer : the generator trainable with a frozen critic attached, and compiled
            critic : the raw cirtic model
            critic_trainer : compiled model, the generator is frozen, but the critic is trainable
    *
    """
    pre_generator, generator = make_generator()
    critic = make_critic_64()
    generator_container = ModelContainer(generator)
    critic_container = ModelContainer(critic)

    # generator pre-training
    pre_generator.compile(optimizer=Adam(LEARNING_RATE, beta_1=0.5, beta_2=0.9),
                          loss=[categorical_crossentropy, categorical_crossentropy])

    L_channel_input_layer = Input(shape=IMAGE_SHAPE_1, name='L_channel_input')
    generated_images_layer = generator_container.apply(L_channel_input_layer, True, 32)
    critic_output_layer = critic_container.apply(generated_images_layer, False)

    # generator training
    generator_trainer = Model(inputs=[L_channel_input_layer], outputs=[critic_output_layer])
    generator_trainer.compile(optimizer=Adam(LEARNING_RATE, beta_1=0.5, beta_2=0.9), loss=weighted_mean_loss)

    # critic training
    L_channel_input_layer = Input(shape=IMAGE_SHAPE_1, name='L_channel_input')
    generated_images_layer = generator_container.apply(L_channel_input_layer, False)
    critic_generated_layer = critic_container.apply(generated_images_layer, True)

    real_images_input_layer = Input(shape=IMAGE_SHAPE_3)
    critic_real_layer = critic_container.apply(real_images_input_layer, True)

    interpolated_layer = Lambda(interpolate_layers,
                                output_shape=IMAGE_SHAPE_3)([real_images_input_layer, generated_images_layer])
    critic_interpolated_layer = critic_container.apply(interpolated_layer, True)
    gp_loss_layer = Lambda(calculate_gradient_penalty,
                           output_shape=(1,), name='gp_loss')([critic_interpolated_layer, interpolated_layer])

    critic_trainer = Model(inputs=[real_images_input_layer, L_channel_input_layer],
                           outputs=[critic_real_layer, critic_generated_layer, gp_loss_layer])

    critic_trainer.compile(optimizer=Adam(LEARNING_RATE, beta_1=0.5, beta_2=0.9), loss=[weighted_mean_loss,
                                                                               weighted_mean_loss, mean_loss])

    return pre_generator, generator, generator_trainer, critic, critic_trainer


def generate_and_compile_only_generator_models():
    # create only model for pre training
    pre_generator, generator = make_generator()

    # generator pre-training
    # use this section for fine-tuning
    """
    for ind, item in enumerate(pre_generator.layers):
        if ind < 44:
            item.trainable = False
        else:
            item.trainable = True
    """
    pre_generator.compile(optimizer=Adam(LEARNING_RATE, beta_1=0.5, beta_2=0.9),
                          loss=[categorical_crossentropy, categorical_crossentropy])

    return pre_generator, generator
