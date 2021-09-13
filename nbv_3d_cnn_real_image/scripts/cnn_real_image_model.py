#!/usr/bin/env python3

#####################################################################################################
# Copyright (c) 2021, Riccardo Monica
#   RIMLab, Department of Engineering and Architecture, University of Parma, Italy
#   http://www.rimlab.ce.unipr.it/
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification, are permitted
# provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions
# and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of
# conditions and the following disclaimer in the documentation and/or other materials provided with
# the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to
# endorse or promote products derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
# IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
# FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
# IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
# OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#####################################################################################################

import tensorflow as tf

import numpy as np
import math
import sys

def get_real_image_model(input_shape, output_mask, max_range):

  kernel_size = 3
  num_layers = 4
  first_layer_channels = 8

  layer_filters = [first_layer_channels*(2**k) for k in range(0, num_layers)]
  print("get_real_image_model: creating network with %d layers and %d first layer channels: %s" %
        (int(num_layers), int(first_layer_channels), str(layer_filters)))

  inputs = tf.keras.layers.Input(shape=input_shape, name='input')
  x = inputs
  
  x = tf.split(x, input_shape[2], axis=3)
  x.append(1 / ((x[0] / max_range) + 1))
  x.append(1 / ((x[5] / max_range) + 1))
  x = tf.concat(x, 3)

  skip_connections = []
  skip_connections_padding = []

  for filters in layer_filters:
    skip_connections.append(x)
    #if input size is not power of two, we need to add 1 to the image size during Conv2Dtranspose below
    #or the output size will differ
    #for some unknown reason, we need to add 1 if the input size IS power of two
    predicted_padding = [1 - (x.get_shape().as_list()[1] % 2), 1 - (x.get_shape().as_list()[2] % 2)]
    skip_connections_padding.append(predicted_padding)
    x = tf.keras.layers.Conv2D(filters=filters,
                               kernel_size=kernel_size,
                               strides=1,
                               activation='tanh',
                               padding='same')(x)
    x = tf.keras.layers.Conv2D(filters=filters,
                               kernel_size=kernel_size,
                               strides=2,
                               activation='tanh',
                               padding='same')(x)

  for i, filters in enumerate(layer_filters[::-1]):
    output_padding = skip_connections_padding[-(i + 1)]
    x = tf.keras.layers.Conv2DTranspose(filters=filters,
                                        kernel_size=kernel_size,
                                        strides=2,
                                        output_padding=output_padding,
                                        activation='tanh',
                                        padding='same')(x)
    sc = skip_connections[-(i + 1)]
    x = tf.keras.layers.concatenate([x, sc], axis=3)

  x = tf.keras.layers.Conv2DTranspose(filters=first_layer_channels,
                                      kernel_size=kernel_size,
                                      strides=1,
                                      activation='tanh',
                                      padding='same')(x)
  x = tf.keras.layers.Conv2DTranspose(filters=first_layer_channels // 2,
                                      kernel_size=kernel_size,
                                      strides=1,
                                      activation='tanh',
                                      padding='same')(x)
  x = tf.keras.layers.Conv2DTranspose(filters=1,
                                      kernel_size=kernel_size,
                                      strides=1,
                                      activation='sigmoid',
                                      padding='same')(x)

  mask = tf.constant(output_mask, dtype='float32', shape=(input_shape[0], input_shape[1], 1))
  outputs = x * mask

  model = tf.keras.models.Model(inputs, outputs, name='cnn_real_image_model')

  model.compile(loss='mse', optimizer='adam')
  #model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=False), optimizer='adam')

  return model
  pass
