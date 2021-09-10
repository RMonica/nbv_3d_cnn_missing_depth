#!/usr/bin/python3

import cnn_real_image_projection_model
import scene_3d_dataset

import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
import PIL
import math
import datetime

import rospy

rospy.init_node('cnn_real_image_autocomplete_generate', anonymous=True)

source_file_name_prefix = rospy.get_param('~source_images_prefix', '')

dest_file_name_prefix = rospy.get_param('~dest_images_prefix', '')

tensorboard_directory = rospy.get_param('~tensorboard_directory', '')

training_dataset_first_element = rospy.get_param('~training_dataset_first_element', 0)
training_dataset_last_element = rospy.get_param('~training_dataset_last_element', 120)
validation_dataset_first_element = rospy.get_param('~validation_dataset_first_element', 120)
validation_dataset_last_element = rospy.get_param('~validation_dataset_last_element', 180)

num_epochs = rospy.get_param('~num_epochs', 300)

load_checkpoint = rospy.get_param('~load_checkpoint', '')

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
    print("cnn_real_image_autocomplete_generate: Exception while limiting GPU memory:")
    print(e)
    exit(1)

final_val_dataset_augmentation = []
batch_size = 1
dataset_augmentation = ['rotation4', ]

dataset, image_width, image_height, image_depth, y_image_width, y_image_height, y_image_depth = (
  scene_3d_dataset.get_scene_3d_dataset(source_file_name_prefix, training_dataset_first_element,
  training_dataset_last_element, dataset_augmentation))
val_dataset, val_image_width, val_image_height, val_image_depth, val_y_image_width, val_y_image_height, val_y_image_depth = (
  scene_3d_dataset.get_scene_3d_dataset(source_file_name_prefix, validation_dataset_first_element,
  validation_dataset_last_element, dataset_augmentation))

dataset.batch(batch_size)
val_dataset.batch(batch_size)

input_shape = (image_depth, image_height, image_width, 2)
batch_size = 1

environment_shape = input_shape
image_shape = (10, 10, 10) # useless
raycasting_downsampling = 1 # useless
initial_values_channels = 2 # useless
max_range = 1.0 # useless
fx = 5 # useless
fy = 5 # useless
cx = 5 # useless
cy = 5 # useless
model = cnn_real_image_projection_model.get_autocomplete_model(environment_shape, image_shape, raycasting_downsampling,
                                                               initial_values_channels,
                                                               max_range, fx, fy, cx, cy)
model_file_prefix = 'autocomplete'
output_channels = 1

model.summary()

if (load_checkpoint != ''):
  rospy.loginfo("cnn_real_image_autocomplete_generate: loading weights from " + dest_file_name_prefix + load_checkpoint)
  model.load_weights(dest_file_name_prefix + load_checkpoint)

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
        self.model.save_weights(dest_file_name_prefix + model_file_prefix +
                                '_epoch_' + str(epoch) + '.chkpt')
      pass
    pass

  str_now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  tensorboard_directory = (tensorboard_directory + str_now + '_' + model_file_prefix)
  tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_directory, histogram_freq=1)

  model.fit(dataset,
            epochs=num_epochs,
            callbacks=[TerminationCallback(), SaveModelCallback(model), tensorboard_callback],
            validation_data=val_dataset
            )

  model_filename = dest_file_name_prefix + model_file_prefix + '_final.chkpt'
  rospy.loginfo("cnn_real_image_autocomplete_generate: saving model '%s'" % model_filename)
  model.save_weights(model_filename)
  pass

rospy.loginfo('cnn_real_image_autocomplete_generate: predicting and saving images...')
counter = 0
dataset, image_width, image_height, image_depth, y_image_width, y_image_height, y_image_depth = (
  scene_3d_dataset.get_scene_3d_dataset(source_file_name_prefix, training_dataset_first_element,
  training_dataset_last_element, final_val_dataset_augmentation))
val_dataset, val_image_width, val_image_height, val_image_depth, val_y_image_width, val_y_image_height, val_y_image_depth = (
  scene_3d_dataset.get_scene_3d_dataset(source_file_name_prefix, validation_dataset_first_element,
  validation_dataset_last_element, final_val_dataset_augmentation))

for i,x in enumerate(dataset.concatenate(val_dataset)):
  img = model.predict(x[0])
  result_filename = dest_file_name_prefix + str(counter) + '_result'

  final_img_shape = [image_depth, image_height, image_width]
  img = np.reshape(img, final_img_shape)

  img = np.clip(img, 0.0, 1.0)

  scene_3d_dataset.save_voxelgrid(result_filename, img)
  
  rospy.loginfo('cnn_real_image_autocomplete_generate: saved image %s' % result_filename)
  
  counter += 1
  pass

rospy.loginfo('cnn_real_image_autocomplete_generate: saved %s images' % str(counter))

