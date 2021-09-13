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

import cnn_real_image_projection_model
import image_and_scene_3d_dataset

import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
import PIL
import math
import datetime

import rospy

rospy.init_node('cnn_real_image_model_generate', anonymous=True)

scenario_file_name_prefix = rospy.get_param("~scenario_file_name_prefix", "")
environment_file_name_prefix = rospy.get_param('~environment_prefix', '')
image_file_name_prefix = rospy.get_param('~image_file_name_prefix', '')
pose_file_name_prefix = rospy.get_param('~pose_file_name_prefix', '')

dest_file_name_prefix = rospy.get_param('~dest_images_prefix', '')

learning_rate = rospy.get_param('~learning_rate', 0.001)

tensorboard_directory = rospy.get_param('~tensorboard_directory', '')

training_dataset_first_element = rospy.get_param('~training_dataset_first_element', 0)
training_dataset_last_element = rospy.get_param('~training_dataset_last_element', 120)
validation_dataset_first_element = rospy.get_param('~validation_dataset_first_element', 120)
validation_dataset_last_element = rospy.get_param('~validation_dataset_last_element', 180)

batch_size = rospy.get_param('~batch_size', 1)

num_epochs = rospy.get_param('~num_epochs', 300)

mask_filename = rospy.get_param('~mask_filename', '')

max_range = rospy.get_param('~max_range', 4.0)

NETWORK_MODE_FULL = "projection"
NETWORK_MODE_ONLY3D = "only3d"
NETWORK_MODE_NOATTENTION = "noattention"
NETWORK_MODE_UNFROZEN = "unfrozen"
network_mode = rospy.get_param('~network_mode', NETWORK_MODE_FULL)

fx = rospy.get_param('~camera_fx', 10)
fy = rospy.get_param('~camera_fy', 10)
cx = rospy.get_param('~camera_cx', 10)
cy = rospy.get_param('~camera_cy', 10)

env_number = rospy.get_param('~partial_environments_number', 5)

raycasting_downsampling = rospy.get_param('~raycasting_downsampling', 16)
image_depth = rospy.get_param('~image_depth', 16)

zero_initial_values = False

load_checkpoint = rospy.get_param('~load_checkpoint', '')
load_partial_checkpoint = rospy.get_param('~load_partial_checkpoint', '')

evaluation_only = rospy.get_param('~evaluation_only', False)

max_memory_mb = rospy.get_param('~max_memory_mb', 3072)
# limit GPU memory
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(
      gpu, [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=max_memory_mb)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
  except RuntimeError as e:
    print("cnn_real_image_model_generate: Exception while limiting GPU memory:")
    print(e)
    exit(1)
    
rospy.loginfo("cnn_real_image_model_generate: loading output_mask %s", mask_filename)
with PIL.Image.open(mask_filename) as im:
  output_mask = im
  im_width, im_height = im.size
  output_mask = np.array(output_mask.getdata(), dtype="float32")
  output_mask = np.reshape(output_mask, (im_height, im_width, 1))

datasets = []
for scenario_num in range(training_dataset_first_element, training_dataset_last_element):
  dataset, image_metadata, scene_3d_metadata = (
    image_and_scene_3d_dataset.get_image_dataset(scenario_file_name_prefix,
                                                 environment_file_name_prefix,
                                                 image_file_name_prefix,
                                                 pose_file_name_prefix,
                                                 scenario_num,
                                                 env_number,
                                                 output_mask,
                                                 zero_initial_values))
  dataset = dataset.batch(batch_size)
  datasets.append(dataset)
  pass

val_datasets = []
for scenario_num in range(validation_dataset_first_element, validation_dataset_last_element):
  val_dataset, val_image_metadata, val_scene_3d_metadata = (
    image_and_scene_3d_dataset.get_image_dataset(scenario_file_name_prefix,
                                                 environment_file_name_prefix,
                                                 image_file_name_prefix,
                                                 pose_file_name_prefix,
                                                 scenario_num,
                                                 env_number,
                                                 output_mask,
                                                 zero_initial_values))
  val_dataset = val_dataset.batch(batch_size)
  val_datasets.append(val_dataset)
  pass

dataset = datasets[0]
for i in range(1, len(datasets)):
  dataset = dataset.concatenate(datasets[i])
val_dataset = val_datasets[0]
for i in range(1, len(val_datasets)):
  val_dataset = val_dataset.concatenate(val_datasets[i])

environment_shape = (scene_3d_metadata['scene_3d_depth'], scene_3d_metadata['scene_3d_height'],
                     scene_3d_metadata['scene_3d_width'], 2)
image_shape = (image_metadata['image_height'],
               image_metadata['image_width'], image_depth)
initial_values_channels = image_metadata['image_channels']

if network_mode == NETWORK_MODE_FULL:
  model = cnn_real_image_projection_model.get_real_image_projection_model(environment_shape,
                                                                          image_shape,
                                                                          raycasting_downsampling,
                                                                          initial_values_channels,
                                                                          max_range, fx, fy, cx, cy,
                                                                          learning_rate=learning_rate)
  model_name = 'projection'
elif network_mode == NETWORK_MODE_UNFROZEN:
  model = cnn_real_image_projection_model.get_real_image_projection_model(environment_shape,
                                                                          image_shape,
                                                                          raycasting_downsampling,
                                                                          initial_values_channels,
                                                                          max_range, fx, fy, cx, cy,
                                                                          learning_rate=learning_rate,
                                                                          freeze_autocomplete_layers=False)
  model_name = 'unfrozen'
elif network_mode == NETWORK_MODE_ONLY3D:
  model = cnn_real_image_projection_model.get_real_image_only3d_model(environment_shape,
                                                                      image_shape,
                                                                      raycasting_downsampling,
                                                                      initial_values_channels,
                                                                      max_range, fx, fy, cx, cy,
                                                                      learning_rate=learning_rate)
  model_name = 'only3d'
elif network_mode == NETWORK_MODE_NOATTENTION:
  model = cnn_real_image_projection_model.get_real_image_noattention_model(environment_shape,
                                                                           image_shape,
                                                                           raycasting_downsampling,
                                                                           initial_values_channels,
                                                                           max_range, fx, fy, cx, cy,
                                                                           learning_rate=learning_rate)
  model_name = 'noattention'
else:
  rospy.logfatal("Unknown network mode: " + network_mode)
  exit(1)

if (load_checkpoint != ''):
  rospy.loginfo("cnn_real_image_model_generate: loading weights from " + load_checkpoint)
  model.load_weights(load_checkpoint)

if (load_partial_checkpoint != ''):
  rospy.loginfo("cnn_real_image_model_generate: loading partial model " + load_partial_checkpoint)
  autocompl_model = cnn_real_image_projection_model.get_autocomplete_model(environment_shape, image_shape,
                                                                           raycasting_downsampling,
                                                                           initial_values_channels,
                                                                           max_range, fx, fy, cx, cy,
                                                                           verbose=False)

  autocompl_model.load_weights(load_partial_checkpoint)
  transferable_layers = cnn_real_image_projection_model.get_transferable_layers()
  rospy.loginfo("cnn_real_image_model_train: transferable layers are " + str(transferable_layers))
  for l in autocompl_model.layers:
    if l.name in transferable_layers:
      for l2 in model.layers:
        if (l2.name == l.name):
          rospy.loginfo("cnn_real_image_projection_train: transferring layer " + l.name)
          l2.set_weights(l.get_weights())
  autocompl_model = None # destroy to save memory
  pass

if (not evaluation_only):

  class TerminationCallback(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
      pass
    def on_epoch_end(self, epoch, logs=None):
      if (rospy.is_shutdown()):
        exit()
      pass
    pass

  class SaveModelCallback(tf.keras.callbacks.Callback):
    def __init__(self, model):
      self.model = model
      pass
    def on_epoch_end(self, epoch, logs=None):
      if (epoch % 20 == 0):
        self.model.save_weights(dest_file_name_prefix + model_name + '_epoch_' + str(epoch) + '.chkpt')
      pass
    pass

  str_now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  tensorboard_directory = (tensorboard_directory + network_mode + "-" + str_now)
  tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_directory, histogram_freq=1)

  model.fit(dataset,
            epochs=num_epochs,
            callbacks=[TerminationCallback(), SaveModelCallback(model), tensorboard_callback],
            validation_data=val_dataset
            )

  model_filename = dest_file_name_prefix + model_name + '_final.chkpt'
  rospy.loginfo("cnn_real_image_model_train: saving model '%s'" % model_filename)
  model.save_weights(model_filename)
  pass

rospy.loginfo('cnn_real_image_model_train: predicting and saving images...')

for sc_n, d in enumerate([*datasets, *val_datasets]):
  env_counter = 0
  counter = 0
  for i, x in enumerate(d):
    batch_img = model.predict(x[0])

    batch_img = np.clip(np.array(batch_img, dtype="float32"), 0.0, 1.0)
    for batch_i in range(0, len(batch_img)):
      result_filename_prefix = (dest_file_name_prefix + str(sc_n + training_dataset_first_element) + "_" +
                                str(counter) + "_" + str(env_counter) + '_result')
    
      img = batch_img[batch_i]
      img = np.transpose(img, [2, 0, 1])
      img = img[0]

      img = (img * 255).astype(np.uint8)
      img = PIL.Image.fromarray(img, 'L') #  'RGB' if color
      result_filename = result_filename_prefix + ".png"
      img.save(result_filename)
    
      rospy.loginfo('cnn_real_image_model_generate: saved image %s' % result_filename)
    
      env_counter += 1
      if (env_counter >= env_number):
        counter += 1
        env_counter = 0
      pass
    pass
  pass

rospy.loginfo('cnn_real_image_model_generate: saved %s images' % str(counter))


