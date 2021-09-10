#!/usr/bin/env python3

import cnn_real_image_model
import image_dataset

import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
import PIL
import math
import datetime

import rospy

rospy.init_node('cnn_real_image_model_generate', anonymous=True)

scenario_file_name_prefix = rospy.get_param('~scenario_file_name_prefix', '')
image_file_name_prefix = rospy.get_param('~image_file_name_prefix', '')

dest_file_name_prefix = rospy.get_param('~dest_images_prefix', '')

model_file_prefix = rospy.get_param('~model_file_prefix', '')

tensorboard_directory = rospy.get_param('~tensorboard_directory', '')

training_dataset_first_element = rospy.get_param('~training_dataset_first_element', 0)
training_dataset_last_element = rospy.get_param('~training_dataset_last_element', 120)
validation_dataset_first_element = rospy.get_param('~validation_dataset_first_element', 120)
validation_dataset_last_element = rospy.get_param('~validation_dataset_last_element', 180)

batch_size = rospy.get_param('~batch_size', 1)

num_epochs = rospy.get_param('~num_epochs', 300)

mask_filename = rospy.get_param('~mask_filename', '')

max_range = rospy.get_param('~max_range', 4.0)

env_number = rospy.get_param('~partial_environments_number', 5)

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
  dataset, image_width, image_height, x_channels, y_image_width, y_image_height = (
    image_dataset.get_image_dataset(scenario_file_name_prefix, image_file_name_prefix, scenario_num, env_number,
    output_mask))
  dataset = dataset.batch(batch_size)
  datasets.append(dataset)
  pass

val_datasets = []
for scenario_num in range(validation_dataset_first_element, validation_dataset_last_element):
  val_dataset, val_image_width, val_image_height, val_x_channels, val_y_image_width, val_y_image_height = (
    image_dataset.get_image_dataset(scenario_file_name_prefix, image_file_name_prefix, scenario_num, env_number,
    output_mask))
  val_dataset = val_dataset.batch(batch_size)
  val_datasets.append(val_dataset)
  pass

dataset = datasets[0]
for i in range(1, len(datasets)):
  dataset = dataset.concatenate(datasets[i])
val_dataset = val_datasets[0]
for i in range(1, len(val_datasets)):
  val_dataset = val_dataset.concatenate(val_datasets[i])

input_shape = (image_height, image_width, x_channels)

model = cnn_real_image_model.get_real_image_model(input_shape, output_mask, max_range)

model.summary()

if (load_checkpoint != ''):
  rospy.loginfo("cnn_real_image_model_generate: loading weights from " + load_checkpoint)
  model.load_weights(load_checkpoint)

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
        self.model.save_weights(model_file_prefix + '_epoch_' + str(epoch) + '.chkpt')
      pass
    pass

  str_now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  tensorboard_directory = (tensorboard_directory + str_now)
  tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_directory, histogram_freq=1)

  model.fit(dataset,
            epochs=num_epochs,
            callbacks=[TerminationCallback(), SaveModelCallback(model), tensorboard_callback],
            validation_data=val_dataset
            )

  model_filename = model_file_prefix + '_final.chkpt'
  rospy.loginfo("cnn_real_image_model_generate: saving model '%s'" % model_filename)
  model.save_weights(model_filename)
  pass

rospy.loginfo('cnn_real_image_model_generate: predicting and saving images...')

for sc_n, d in enumerate([*datasets, *val_datasets]):
  env_counter = 0
  counter = 0
  for i,x in enumerate(d):
    batch_img = model.predict(x[0])
    
    batch_img = np.clip(np.array(batch_img, dtype="float32"), 0.0, 1.0)
    for batch_i in range(0, len(batch_img)):   
      img = batch_img[batch_i]
      
      result_filename = (dest_file_name_prefix + str(sc_n + training_dataset_first_element) + "_" +
                         str(counter) + "_" + str(env_counter) + '_result')

      img = np.transpose(img, [2, 0, 1])
      img = img[0]
      img = (img * 255).astype(np.uint8)
      img = PIL.Image.fromarray(img, 'L') #  'RGB' if color
      img.save(result_filename + ".png")
    
      rospy.loginfo('cnn_real_image_model_generate: saved image %s' % result_filename)
    
      env_counter += 1
      if (env_counter >= env_number):
        counter += 1
        env_counter = 0
      pass
    pass
  pass

rospy.loginfo('cnn_real_image_model_generate: saved %s images' % str(counter))


