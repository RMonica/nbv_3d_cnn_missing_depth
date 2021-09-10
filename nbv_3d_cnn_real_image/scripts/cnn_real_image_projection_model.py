#!/usr/bin/env python3

import tensorflow as tf

import numpy as np
import math
import sys

import rospy

def generate_default_pose_indices(image_shape, max_range, fx, fy, cx, cy, verbose=True):

  image_height = image_shape[0]
  image_width = image_shape[1]
  image_depth = image_shape[2]
  
  default_pose_indices = []

  for yi in range(0, image_height):
    default_pose_indices_y = []
    for xi in range(0, image_width):
      default_pose_indices_x = []
      for zi in range(0, image_depth):
        z = zi * (max_range / image_depth)
        x = (xi - cx + 0.5) / fx * z
        y = (yi - cy + 0.5) / fy * z
        default_pose_indices_x.append([x, y, z, ])
      default_pose_indices_y.append(default_pose_indices_x)
    default_pose_indices.append(default_pose_indices_y)
  default_pose_indices = np.array(default_pose_indices, dtype="float32")

  return default_pose_indices
  
def build_autocomplete_network(name_prefix, environment, autocompleted_environment_kernel_size, channels_out,
                               verbose=True, freeze_first_layers=False):

  kernel_size = autocompleted_environment_kernel_size

  num_layers = 3
  layer_filters = [16*(2**k) for k in range(0, num_layers)]
  if verbose:
    rospy.loginfo("  build_autocomplete_network: creating network with %d layers: %s" %
                  (int(num_layers), str(layer_filters)))
  x = environment

  skip_connections = []
  skip_connections_padding = []

  for i, filters in enumerate(layer_filters):
    skip_connections.append(x)
    #if input size is not power of two, we need to add 1 to the image size during Conv3Dtranspose below
    #or the output size will differ
    #for some unknown reason, we need to add 1 if the input size IS power of two
    predicted_padding = [1 - (x.get_shape().as_list()[1] % 2), 1 - (x.get_shape().as_list()[2] % 2),
                         1 - (x.get_shape().as_list()[3] % 2)]
    skip_connections_padding.append(predicted_padding)
    x = tf.keras.layers.Conv3D(filters=filters,
                               kernel_size=kernel_size,
                               strides=1,
                               activation='linear',
                               padding='same',
                               name=(name_prefix + "ac_e1_" + str(i)))(x)
    x = tf.keras.layers.BatchNormalization(name=("ac_bn1_" + str(i)))(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Conv3D(filters=filters,
                               kernel_size=kernel_size,
                               strides=2,
                               activation='linear',
                               padding='same',
                               name=(name_prefix + "ac_e2_" + str(i)))(x)
    x = tf.keras.layers.BatchNormalization(name=("ac_bn2_" + str(i)))(x)
    x = tf.keras.layers.LeakyReLU()(x)
    if verbose:
      rospy.loginfo("  build_autocomplete_network: created layer %u" % i)

  for i, filters in enumerate(layer_filters[::-1]):
    output_padding = skip_connections_padding[-(i + 1)]
    x = tf.keras.layers.Conv3DTranspose(filters=filters,
                                        kernel_size=kernel_size,
                                        strides=2,
                                        output_padding=output_padding,
                                        activation='linear',
                                        padding='same',
                                        name=(name_prefix + "ac_d1_" + str(i)))(x)
    x = tf.keras.layers.BatchNormalization(name=("ac_bn3_" + str(i)))(x)
    x = tf.keras.layers.LeakyReLU()(x)
    sc = skip_connections[-(i + 1)]
    x = tf.keras.layers.concatenate([x, sc], axis=4)
    if verbose:
      rospy.loginfo("  build_autocomplete_network: created reverse layer %u" % i)

  stopped_x = x
  if freeze_first_layers:
    stopped_x = tf.stop_gradient(stopped_x)

  autocompleted_environment = tf.keras.layers.Conv3DTranspose(filters=(channels_out - 1),
                                                   kernel_size=1,
                                                   strides=1,
                                                   activation='linear',
                                                   padding='same',
                                                   name=(name_prefix + "ac_f"))(stopped_x)
                                                   
  occupancy_prob = tf.keras.layers.Conv3DTranspose(filters=1,
                                                   kernel_size=1,
                                                   strides=1,
                                                   activation='sigmoid',
                                                   padding='same',
                                                   name=(name_prefix + "ac_p"))(x)
  if freeze_first_layers:
    occupancy_prob = tf.stop_gradient(occupancy_prob)

  autocompleted_environment = tf.concat([autocompleted_environment, occupancy_prob], axis=4)

  if verbose:
    rospy.loginfo("  build_autocomplete_network: autocomplete network created, output shape is %s." %
                  str(autocompleted_environment.shape))

  return autocompleted_environment, occupancy_prob

# returns output_channels * slice_channels channels
def generate_self_attention_network(name_prefix, max_range, slice,
                                    num_layers,
                                    output_channels,
                                    verbose=True):
  if verbose:
    rospy.loginfo("  generate_self_attention_network: creating self-attention network.")
    rospy.loginfo("  generate_self_attention_network: input shape is %s." % str(slice.shape))
    rospy.loginfo("  generate_self_attention_network: output channels is %u." % output_channels)

  kernel_size = 3
  slice_shape = slice.shape
  slice_channels = slice_shape[3]
  slice_length = slice_shape[2]
  slice_slices = slice_shape[1]

  prev_attention = tf.fill(tf.shape(slice), 0.0)

  increasing_by_z_coord = tf.constant(np.array(range(0, slice_length)) / (slice_length - 1) * max_range, dtype="float32")
  increasing_by_z_coord = tf.expand_dims(increasing_by_z_coord, 0)
  increasing_by_z_coord = tf.repeat(increasing_by_z_coord, slice_slices, axis=0)
  increasing_by_z_coord = tf.expand_dims(increasing_by_z_coord, 0)
  increasing_by_z_coord = tf.repeat(increasing_by_z_coord, tf.shape(slice)[0], axis=0)
  increasing_by_z_coord = tf.expand_dims(increasing_by_z_coord, -1)

  slice_plus_z = tf.concat([slice, increasing_by_z_coord], axis=3)
  slice_plus_z_channels = slice_channels + 1

  ray_summary = []
  saliency_summary = []

  for c in range(0, output_channels):
    x = slice_plus_z
    for i in range(0, num_layers):
      x = tf.concat([x, prev_attention], axis=3)
      x = tf.keras.layers.Conv1D(filters=slice_plus_z_channels,
                                 kernel_size=kernel_size,
                                 strides=1,
                                 input_shape=(slice_length, slice_plus_z_channels + 1),
                                 activation='linear',
                                 padding='same',
                                 name=(name_prefix + "at_e_ch" + str(c) + "_l" + str(i)))(x)
      x = tf.keras.layers.LeakyReLU()(x)
      if verbose:
        rospy.loginfo("  generate_self_attention_network: creating layer %u." % i)
      pass

    x = tf.concat([x, prev_attention], axis=3)
    x = tf.keras.layers.Conv1D(filters=1,
                               kernel_size=kernel_size,
                               strides=1,
                               input_shape=(slice_length, slice_plus_z_channels + 1),
                               activation='linear',
                               padding='same',
                               name=(name_prefix + "at_f_ch" + str(c)))(x)
    if verbose:
      rospy.loginfo("  generate_self_attention_network: created final layer, shape is %s." % str(x.shape))

    x = tf.nn.softmax(x, axis=2)
    saliency_summary.append(x)
    dup_x = [x for i in range(0, slice_plus_z_channels)]
    dup_x = tf.concat(dup_x, axis=3)

    if verbose:
      rospy.loginfo("  generate_self_attention_network: created saliency layer %s." % str(dup_x.shape))

    summary = dup_x * slice_plus_z
    summary = tf.math.reduce_sum(summary, axis=2, keepdims=False, name=None)
    if verbose:
      rospy.loginfo("  generate_self_attention_network: created ray summary %s." % str(summary.shape))
    ray_summary.append(summary)
    prev_attention = prev_attention + x
    pass

  ray_summary = tf.concat(ray_summary, axis=2)
  saliency_summary = tf.concat(saliency_summary, axis=3)

  if verbose:
    rospy.loginfo("  generate_self_attention_network: self-attention network created, output shape is %s." %
                  str(ray_summary.shape))

  return ray_summary, saliency_summary

def generate_real_image_network(name_prefix, inputs, initial_values, mask,
                                num_layers,
                                verbose=True):
  if verbose:
    rospy.loginfo("  generate_real_image_network: creating real image network.")
    rospy.loginfo("  generate_real_image_network: input shape is %s." % str(inputs.shape))
    rospy.loginfo("  generate_real_image_network: initial values shape is %s." % str(initial_values.shape))
    rospy.loginfo("  generate_real_image_network: mask has shape %s.", str(mask.shape))
    pass

  kernel_size = 3
  first_layer_channels = 8

  layer_filters = [first_layer_channels*(2**k) for k in range(0, num_layers)]

  initial_values_padded = initial_values

  if (num_layers != 0):
    max_sensor_range = 4.0
    initial_values_padded = tf.split(initial_values, initial_values.shape[3], axis=3)
    initial_values_padded.append(1 / ((initial_values_padded[0] / max_sensor_range) + 1))
    initial_values_padded.append(1 / ((initial_values_padded[5] / max_sensor_range) + 1))
    initial_values_padded = tf.concat(initial_values_padded, 3)
    pass

  x = tf.concat([inputs, initial_values_padded], axis=3)

  skip_connections = []
  skip_connections_padding = []

  for i, filters in enumerate(layer_filters):
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
                               padding='same',
                               name=(name_prefix + "ri_e1_" + str(i)))(x)
    x = tf.keras.layers.Conv2D(filters=filters,
                               kernel_size=kernel_size,
                               strides=2,
                               activation='tanh',
                               padding='same',
                               name=(name_prefix + "ri_e2_" + str(i)))(x)

  for i, filters in enumerate(layer_filters[::-1]):
    output_padding = skip_connections_padding[-(i + 1)]
    x = tf.keras.layers.Conv2DTranspose(filters=filters,
                                        kernel_size=kernel_size,
                                        strides=2,
                                        output_padding=output_padding,
                                        activation='tanh',
                                        padding='same',
                                        name=(name_prefix + "ri_d1_" + str(i)))(x)
    sc = skip_connections[-(i + 1)]
    x = tf.keras.layers.concatenate([x, sc], axis=3)

  x = tf.keras.layers.Conv2DTranspose(filters=(1 if num_layers == 0 else first_layer_channels),
                                      kernel_size=kernel_size,
                                      strides=1,
                                      activation='tanh',
                                      padding='same',
                                      name=(name_prefix + "ri_f1"))(x)
  x = tf.keras.layers.Conv2DTranspose(filters=(1 if num_layers == 0 else first_layer_channels // 2),
                                      kernel_size=kernel_size,
                                      strides=1,
                                      activation='tanh',
                                      padding='same',
                                      name=(name_prefix + "ri_f2"))(x)
  x = tf.keras.layers.Conv2DTranspose(filters=1,
                                      kernel_size=kernel_size,
                                      strides=1,
                                      activation='sigmoid',
                                      padding='same',
                                      name=(name_prefix + "ri_f3"))(x)

  outputs = x * mask

  if verbose:
    rospy.loginfo("  generate_real_image_network: real image network created, output shape is %s.", str(outputs.shape))

  return outputs
  pass

def build_real_image_projection_model(name_prefix,
                                      environment,
                                      input_pose_rotation,
                                      input_pose_translation,
                                      input_output_mask,
                                      input_initial_values,
                                      environment_shape, image_shape, raycasting_downsampling,
                                      initial_values_channels,
                                      max_range, fx, fy, cx, cy,
                                      verbose=True,
                                      freeze_autocomplete_layers=False,
                                      skip_attention=False):

  # pre-generate sampling indices
  raycasting_image_shape = (image_shape[0] // raycasting_downsampling, image_shape[1] // raycasting_downsampling,
                            image_shape[2], )
  default_pose_indices = generate_default_pose_indices(raycasting_image_shape, max_range, fx / raycasting_downsampling,
    fy / raycasting_downsampling, cx / raycasting_downsampling, cy / raycasting_downsampling,
    verbose=verbose)
  default_pose_indices = tf.constant(default_pose_indices, dtype='float32', shape=[raycasting_image_shape[0],
                                                                                   raycasting_image_shape[1],
                                                                                   raycasting_image_shape[2], 3])

  if verbose:
    rospy.loginfo("cnn_real_image_projection_model: creating model.")
  
  # ---------- INPUTS --------------
  output_mask = input_output_mask
  initial_values = input_initial_values
  rospy.loginfo("cnn_real_image_projection_model: input 0 has shape %s" % str(environment.shape))
  rospy.loginfo("cnn_real_image_projection_model: input 1 has shape %s" % str(input_pose_rotation.shape))
  rospy.loginfo("cnn_real_image_projection_model: input 2 has shape %s" % str(input_pose_translation.shape))
  rospy.loginfo("cnn_real_image_projection_model: input 3 has shape %s" % str(output_mask.shape))
  rospy.loginfo("cnn_real_image_projection_model: input 4 has shape %s" % str(initial_values.shape))
  
  # --------- AUTOCOMPLETE NETWORK -----------
  
  autocompleted_environment_channels = 4
  autocompleted_environment_kernel_size = 3
  autocompleted_environment, occupancy_prob = build_autocomplete_network(name_prefix, environment,
                                                                         autocompleted_environment_kernel_size,
                                                                         autocompleted_environment_channels,
                                                                         verbose=verbose,
                                                                         freeze_first_layers=freeze_autocomplete_layers)
  
  # ---------- PROJECTION TRANSFORMATION ---------
  
  if verbose:
    rospy.loginfo("  projection_transformation: creating projection transformation.")
  flattened_pose_indices = tf.reshape(default_pose_indices, (1, raycasting_image_shape[0] * raycasting_image_shape[1] *
                                                                raycasting_image_shape[2], 3))
  flattened_pose_indices = tf.repeat(flattened_pose_indices, repeats=tf.shape(input_pose_rotation)[0], axis=0)
  flattened_pose_indices = tf.transpose(flattened_pose_indices, [0, 2, 1])
  flattened_pose_indices = tf.matmul(input_pose_rotation, flattened_pose_indices)
  flattened_pose_indices = tf.transpose(flattened_pose_indices, [0, 2, 1])
  flattened_pose_indices = tf.transpose(flattened_pose_indices, [1, 0, 2]) # move batch size in the middle
  flattened_pose_indices = flattened_pose_indices + input_pose_translation
  flattened_pose_indices = tf.transpose(flattened_pose_indices, [1, 0, 2]) # move batch size back
  # reorder indices so that z is first
  flattened_pose_indices = tf.transpose(flattened_pose_indices, [2, 0, 1])
  flattened_pose_indices = [flattened_pose_indices[2], flattened_pose_indices[1], flattened_pose_indices[0], ]
  flattened_pose_indices = tf.transpose(flattened_pose_indices, [1, 2, 0])
  flattened_pose_indices = tf.cast(flattened_pose_indices, dtype='int32')
  projected_env = tf.gather_nd(autocompleted_environment, flattened_pose_indices, batch_dims=1)
  projected_env = tf.reshape(projected_env, (-1, raycasting_image_shape[0], raycasting_image_shape[1],
                                                 raycasting_image_shape[2],
                                                 autocompleted_environment_channels))
  if verbose:
    rospy.loginfo("  projection_transformation: projection transformation created, output shape is %s." %
                  str(projected_env.shape))

  # --------- SELF-ATTENTION -----------

  if not skip_attention:
    attention_channels = 8
    attention_num_layers = 1

    sliced_rays = tf.reshape(projected_env, (-1, raycasting_image_shape[0] * raycasting_image_shape[1],
                                             raycasting_image_shape[2], autocompleted_environment_channels))
    rays_image, saliency_image = generate_self_attention_network(name_prefix, max_range, sliced_rays,
                                                                 attention_num_layers, attention_channels,
                                                                 verbose=verbose)
    rays_image = tf.reshape(rays_image, (-1, raycasting_image_shape[0], raycasting_image_shape[1],
                                         (autocompleted_environment_channels + 1) * attention_channels))
    saliency_image = tf.reshape(saliency_image, (-1, raycasting_image_shape[0], raycasting_image_shape[1],
                                                 raycasting_image_shape[2],
                                                 attention_channels))
  else:
    saliency_image = []
    rays_image = tf.reshape(projected_env, (-1, raycasting_image_shape[0], raycasting_image_shape[1],
                                           autocompleted_environment_channels * raycasting_image_shape[2]))

  # --------- UPSAMPLING ---------------

  if verbose:
    rospy.loginfo("  upsampling: input shape is %s." % str(rays_image.shape))
  rays_image = tf.keras.layers.UpSampling2D(size=(raycasting_downsampling, raycasting_downsampling),
                                            interpolation='bilinear')(rays_image)
  if verbose:
    rospy.loginfo("  upsampling: output shape is %s." % str(rays_image.shape))
  
  # --------- IMAGE-SPACE PREDICTION ---

  real_image_network_num_layers = 4
  predicted_image = generate_real_image_network(name_prefix, rays_image, initial_values, output_mask,
                                                real_image_network_num_layers, verbose=verbose)
  
  return predicted_image, occupancy_prob, saliency_image

def get_real_image_projection_model(environment_shape, image_shape, raycasting_downsampling,
                                    initial_values_channels,
                                    max_range, fx, fy, cx, cy, verbose=True,
                                    learning_rate=0.001,
                                    freeze_autocomplete_layers=True):

  if verbose:
    rospy.loginfo("cnn_real_image_projection_model: creating model.")

  environment = tf.keras.layers.Input(shape=environment_shape, name='input_environment', dtype='float32')
  input_pose_rotation = tf.keras.layers.Input(shape=(3, 3), name='input_pose_rotation', dtype='float32')
  input_pose_translation = tf.keras.layers.Input(shape=(3, ), name='input_pose_translation', dtype='float32')
  input_output_mask = tf.keras.layers.Input(shape=(image_shape[0], image_shape[1], 1), name='input_output_mask',
                                            dtype='float32')
  input_initial_values = tf.keras.layers.Input(shape=(image_shape[0], image_shape[1], initial_values_channels),
                                                name='input_initial_values', dtype='float32')

  name_prefix = ""
  predicted_image, occupancy_prob, saliency_image = build_real_image_projection_model(name_prefix,
                                                                      environment,
                                                                      input_pose_rotation,
                                                                      input_pose_translation,
                                                                      input_output_mask,
                                                                      input_initial_values,
                                                                      environment_shape, image_shape, raycasting_downsampling,
                                                                      initial_values_channels,
                                                                      max_range, fx, fy, cx, cy,
                                                                      verbose=verbose,
                                                                      freeze_autocomplete_layers=freeze_autocomplete_layers)

  model = tf.keras.Model(inputs=[environment, input_pose_rotation, input_pose_translation, input_output_mask,
                                 input_initial_values], outputs=predicted_image)

  optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

  model.compile(loss='mse', optimizer=optimizer)
  if verbose:
    model.summary()

  rospy.loginfo("cnn_real_image_projection_model: model created.")

  return model

def get_real_image_projection_with_saliency_model(environment_shape, image_shape, raycasting_downsampling,
                                                  initial_values_channels,
                                                  max_range, fx, fy, cx, cy, verbose=True,
                                                  learning_rate=0.001,
                                                  freeze_autocomplete_layers=True):

  if verbose:
    rospy.loginfo("cnn_real_image_projection_model: creating model.")

  environment = tf.keras.layers.Input(shape=environment_shape, name='input_environment', dtype='float32')
  input_pose_rotation = tf.keras.layers.Input(shape=(3, 3), name='input_pose_rotation', dtype='float32')
  input_pose_translation = tf.keras.layers.Input(shape=(3, ), name='input_pose_translation', dtype='float32')
  input_output_mask = tf.keras.layers.Input(shape=(image_shape[0], image_shape[1], 1), name='input_output_mask',
                                            dtype='float32')
  input_initial_values = tf.keras.layers.Input(shape=(image_shape[0], image_shape[1], initial_values_channels),
                                                name='input_initial_values', dtype='float32')

  name_prefix = ""
  predicted_image, occupancy_prob, saliency_image = build_real_image_projection_model(name_prefix,
                                                                      environment,
                                                                      input_pose_rotation,
                                                                      input_pose_translation,
                                                                      input_output_mask,
                                                                      input_initial_values,
                                                                      environment_shape, image_shape, raycasting_downsampling,
                                                                      initial_values_channels,
                                                                      max_range, fx, fy, cx, cy,
                                                                      verbose=verbose,
                                                                      freeze_autocomplete_layers=freeze_autocomplete_layers)

  model = tf.keras.Model(inputs=[environment, input_pose_rotation, input_pose_translation, input_output_mask,
                                 input_initial_values], outputs=[predicted_image, saliency_image])

  optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

  model.compile(loss='mse', optimizer=optimizer)
  if verbose:
    model.summary()

  rospy.loginfo("cnn_real_image_projection_with_saliency_model: model created.")

  return model

def get_real_image_only3d_with_saliency_model(environment_shape, image_shape, raycasting_downsampling,
                                              initial_values_channels,
                                              max_range, fx, fy, cx, cy, verbose=True,
                                              learning_rate=0.001):

  if verbose:
    rospy.loginfo("cnn_real_image_only3d_with_saliency_model: creating model.")

  environment = tf.keras.layers.Input(shape=environment_shape, name='input_environment', dtype='float32')
  input_pose_rotation = tf.keras.layers.Input(shape=(3, 3), name='input_pose_rotation', dtype='float32')
  input_pose_translation = tf.keras.layers.Input(shape=(3, ), name='input_pose_translation', dtype='float32')
  input_output_mask = tf.keras.layers.Input(shape=(image_shape[0], image_shape[1], 1),
                                            name='input_output_mask',
                                            dtype='float32')
  input_initial_values = tf.zeros((tf.shape(environment)[0], image_shape[0], image_shape[1], initial_values_channels, ),
                                  name='input_initial_values', dtype='float32')

  name_prefix = ""
  predicted_image, occupancy_prob, saliency_image = build_real_image_projection_model(name_prefix,
                                                                      environment,
                                                                      input_pose_rotation,
                                                                      input_pose_translation,
                                                                      input_output_mask,
                                                                      input_initial_values,
                                                                      environment_shape, image_shape, raycasting_downsampling,
                                                                      initial_values_channels,
                                                                      max_range, fx, fy, cx, cy,
                                                                      verbose=verbose,
                                                                      freeze_autocomplete_layers=True)

  model = tf.keras.Model(inputs=[environment, input_pose_rotation, input_pose_translation, input_output_mask, ],
                         outputs=[predicted_image, saliency_image])

  optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

  model.compile(loss='mse', optimizer=optimizer)
  if verbose:
    model.summary()

  rospy.loginfo("cnn_real_image_only3d_with_saliency_model: model created.")

  return model

def get_real_image_only3d_model(environment_shape, image_shape, raycasting_downsampling,
                                initial_values_channels,
                                max_range, fx, fy, cx, cy, verbose=True,
                                learning_rate=0.001):

  if verbose:
    rospy.loginfo("cnn_real_image_only3d_model: creating model.")

  environment = tf.keras.layers.Input(shape=environment_shape, name='input_environment', dtype='float32')
  input_pose_rotation = tf.keras.layers.Input(shape=(3, 3), name='input_pose_rotation', dtype='float32')
  input_pose_translation = tf.keras.layers.Input(shape=(3, ), name='input_pose_translation', dtype='float32')
  input_output_mask = tf.keras.layers.Input(shape=(image_shape[0], image_shape[1], 1), name='input_output_mask',
                                            dtype='float32')
  input_initial_values = tf.zeros((tf.shape(environment)[0], image_shape[0], image_shape[1], initial_values_channels, ),
                                  name='input_initial_values', dtype='float32')

  name_prefix = ""
  predicted_image, occupancy_prob, saliency_image = build_real_image_projection_model(name_prefix,
                                                                      environment,
                                                                      input_pose_rotation,
                                                                      input_pose_translation,
                                                                      input_output_mask,
                                                                      input_initial_values,
                                                                      environment_shape, image_shape, raycasting_downsampling,
                                                                      initial_values_channels,
                                                                      max_range, fx, fy, cx, cy,
                                                                      verbose=verbose,
                                                                      freeze_autocomplete_layers=True)

  model = tf.keras.Model(inputs=[environment, input_pose_rotation, input_pose_translation, input_output_mask, ],
                         outputs=predicted_image)

  optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

  model.compile(loss='mse', optimizer=optimizer)
  if verbose:
    model.summary()

  rospy.loginfo("cnn_real_image_only3d_model: model created.")

  return model

def get_real_image_noattention_model(environment_shape, image_shape, raycasting_downsampling,
                                    initial_values_channels,
                                    max_range, fx, fy, cx, cy, verbose=True,
                                    learning_rate=0.001):

  if verbose:
    rospy.loginfo("cnn_real_image_noattention_model: creating model.")

  environment = tf.keras.layers.Input(shape=environment_shape, name='input_environment', dtype='float32')
  input_pose_rotation = tf.keras.layers.Input(shape=(3, 3), name='input_pose_rotation', dtype='float32')
  input_pose_translation = tf.keras.layers.Input(shape=(3, ), name='input_pose_translation', dtype='float32')
  input_output_mask = tf.keras.layers.Input(shape=(image_shape[0], image_shape[1], 1), name='input_output_mask',
                                            dtype='float32')
  input_initial_values = tf.keras.layers.Input(shape=(image_shape[0], image_shape[1], initial_values_channels),
                                                name='input_initial_values', dtype='float32')

  name_prefix = ""
  predicted_image, occupancy_prob, saliency_image = build_real_image_projection_model(name_prefix,
                                                                      environment,
                                                                      input_pose_rotation,
                                                                      input_pose_translation,
                                                                      input_output_mask,
                                                                      input_initial_values,
                                                                      environment_shape, image_shape, raycasting_downsampling,
                                                                      initial_values_channels,
                                                                      max_range, fx, fy, cx, cy,
                                                                      verbose=verbose,
                                                                      freeze_autocomplete_layers=True,
                                                                      skip_attention=True)

  model = tf.keras.Model(inputs=[environment, input_pose_rotation, input_pose_translation, input_output_mask,
                                 input_initial_values], outputs=predicted_image)

  optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

  model.compile(loss='mse', optimizer=optimizer)
  if verbose:
    model.summary()

  rospy.loginfo("cnn_real_image_noattention_model: model created.")

  return model

def get_autocomplete_model(environment_shape, image_shape, raycasting_downsampling,
                           initial_values_channels,
                           max_range, fx, fy, cx, cy, verbose=True,
                           learning_rate=0.001):

  rospy.loginfo("cnn_real_image_projection_model: creating autocomplete model.")

  environment = tf.keras.layers.Input(shape=environment_shape, name='input_environment', dtype='float32')
  input_pose_rotation = tf.ones(shape=(tf.shape(environment)[0], 3, 3), name='input_pose_rotation', dtype='float32')
  input_pose_translation = tf.ones(shape=(tf.shape(environment)[0], 3, ), name='input_pose_translation', dtype='float32')
  input_output_mask = tf.ones(shape=(tf.shape(environment)[0], image_shape[0], image_shape[1], 1), name='input_output_mask',
                              dtype='float32')
  input_initial_values = tf.ones(shape=(tf.shape(environment)[0], image_shape[0], image_shape[1], initial_values_channels),
                                                name='input_initial_values', dtype='float32')

  name_prefix = ""
  predicted_image, occupancy_prob, saliency_image = build_real_image_projection_model(
                                                                 name_prefix,
                                                                 environment,
                                                                 input_pose_rotation,
                                                                 input_pose_translation,
                                                                 input_output_mask,
                                                                 input_initial_values,
                                                                 environment_shape,
                                                                 image_shape,
                                                                 raycasting_downsampling,
                                                                 initial_values_channels,
                                                                 max_range, fx, fy, cx, cy,
                                                                 verbose=verbose,
                                                                 freeze_autocomplete_layers=False)

  model = tf.keras.Model(inputs=[environment, ], outputs=occupancy_prob)
  optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
  model.compile(loss='mse', optimizer=optimizer)
  if verbose:
    model.summary()

  rospy.loginfo("cnn_real_image_projection_model: autocomplete model created.")

  return model

def get_transferable_layers():
  layers_names = []
  # autocomplete layers
  layers_names.extend(["ac_e1_0", "ac_e2_0", "ac_e1_1", "ac_e2_1", "ac_e1_2", "ac_e2_2",
                       "ac_d1_0", "ac_d1_1", "ac_d1_2", 'ac_p', 'ac_f'])
  #autocomplete batch norm
  layers_names.extend(["ac_bn1_0", "ac_bn1_1", "ac_bn1_2", "ac_bn2_0", "ac_bn2_1", "ac_bn2_2",
                       "ac_bn3_0", "ac_bn3_1", "ac_bn3_2"])
  # 3d convolution layers
  layers_names.extend(["i3c_0", "i3c_1"])
  # attention layers
  for c in range(0, 8):
    layers_names.append("at_f_ch")
    for i in range(0, 2):
      layers_names.append("at_e_ch" + str(c) + "_l" + str(i))

  return layers_names
