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
import matplotlib.pyplot as plt
import cv2
import math
import random

import rospy

positional_channels_cache = {}
image_cache_16bit = {}
image_cache_color = {}
image_cache_8bit = {}
def check_image_cache_max_size():
  global image_cache_16bit
  global image_cache_color
  global image_cache_8bit
  IMAGE_CACHE_MAX_SIZE = 7 * 1000
  total_len = len(image_cache_16bit) + len(image_cache_color) + len(image_cache_8bit)
  if total_len >= IMAGE_CACHE_MAX_SIZE:
    return False
  return True

def load_image_16bit(infilename):
  img = cv2.imread(infilename, cv2.IMREAD_UNCHANGED)
  if (img is None):
    rospy.logerr("cnn_real_image_model_generate: could not load image %s" % infilename)
    raise IOError()
  data = np.asarray(img)
  data = data.astype('float32') / 1000.0
  return data

def load_image_16bit_cached(infilename):
  if infilename in image_cache_16bit:
    return image_cache_16bit[infilename]
  else:
    img = load_image_16bit(infilename)
    if check_image_cache_max_size():
      image_cache_16bit[infilename] = img
    return img
  
def load_image_8bit(infilename):
  img = cv2.imread(infilename, cv2.IMREAD_UNCHANGED)
  if (img is None):
    rospy.logerr("cnn_real_image_model_generate: could not load image %s" % infilename)
    raise IOError()
  data = np.asarray(img)
  data = data.astype('float32') / 255.0
  return data

def load_image_8bit_cached(infilename):
  if infilename in image_cache_8bit:
    return image_cache_8bit[infilename]
  else:
    img = load_image_8bit(infilename)
    if check_image_cache_max_size():
      image_cache_8bit[infilename] = img
    return img
  
def load_image_color(infilename):
  img = cv2.imread(infilename, cv2.IMREAD_UNCHANGED)
  if (img is None):
    rospy.logerr("cnn_real_image_model_generate: could not load image %s" % infilename)
    raise IOError()
  data = np.asarray(img)
  data = data.astype('float32') / 255.0
  return data

def load_image_color_cached(infilename):
  if infilename in image_cache_color:
    return image_cache_color[infilename]
  else:
    img = load_image_color(infilename)
    if check_image_cache_max_size():
      image_cache_color[infilename] = img
    return img

def get_image_next_kernel(scenario_file_name_prefix,
                          image_file_name_prefix,
                          output_mask,
                          scenario_num, env_number):
  image_load_ok = True
  view_count = 0
  env_count = 0
  scenario_count = 0

  global positional_channels_cache

  while (scenario_count < 1):
    scenario_prefix = scenario_file_name_prefix + str(scenario_num)
    image_prefix = scenario_prefix + image_file_name_prefix
    depth_filename = image_prefix + "raw_depth_" + str(env_count) + "_" + str(view_count) + ".png"
    depth_normal_filename = image_prefix + "raw_normal_" + str(env_count) + "_" + str(view_count) + ".png"
    robot_filename = image_prefix + "raw_robot_" + str(env_count) + "_" + str(view_count) + ".png"
    robot_normal_filename = image_prefix + "raw_robot_normal_" + str(env_count) + "_" + str(view_count) + ".png"
    output_filename = image_prefix + "gt_" + str(view_count) + ".png"

    try:
      #rospy.loginfo("cnn_real_image_model_generate: loading depth_image '%s'" % depth_filename)
      depth_image = load_image_16bit_cached(depth_filename)
      #rospy.loginfo("  shape is %s" % str(depth_image.shape))
      #rospy.loginfo("cnn_real_image_model_generate: loading depth_normal_filename '%s'" % depth_normal_filename)
      depth_normal_image = load_image_color_cached(depth_normal_filename)
      #rospy.loginfo("  shape is %s" % str(depth_normal_image.shape))
      #rospy.loginfo("cnn_real_image_model_generate: loading robot_filename '%s'" % robot_filename)
      robot_image = load_image_color_cached(robot_filename)
      #rospy.loginfo("  shape is %s" % str(robot_image.shape))
      #rospy.loginfo("cnn_real_image_model_generate: loading robot_normal_filename '%s'" % robot_normal_filename)
      robot_normal_image = load_image_color_cached(robot_normal_filename)
      #rospy.loginfo("  shape is %s" % str(robot_normal_image.shape))
      
      #rospy.loginfo("cnn_real_image_model_generate: loading output_image '%s'" % output_filename)
      output_image = load_image_8bit_cached(output_filename)
      output_image = [output_image, ]
      output_image = np.transpose(output_image, [1, 2, 0])
      
      depth_normal_image = depth_normal_image * 2.0 - 1.0
      robot_normal_image = robot_normal_image * 2.0 - 1.0
      
      depth_image_greater_than_zero = np.array(np.greater(depth_image, 0.0), dtype="float32")
      robot_image_greater_than_zero = np.array(np.greater(robot_image, 0.0), dtype="float32")
      
      depth_image_shape = depth_image.shape
      depth_image_shape_key = str(depth_image_shape)
      if depth_image_shape_key in positional_channels_cache:
        positional_channel_x = positional_channels_cache[depth_image_shape_key][0]
        positional_channel_y = positional_channels_cache[depth_image_shape_key][1]
      else:
        positional_channel_x = [list(range(0, depth_image_shape[1])) for i in range(0, depth_image_shape[0])]
        positional_channel_x = np.array(positional_channel_x).astype('float32') / float(depth_image_shape[1])
        positional_channel_y = [list(range(0, depth_image_shape[0])) for i in range(0, depth_image_shape[1])]
        positional_channel_y = np.array(positional_channel_y).astype('float32') / float(depth_image_shape[0])
        positional_channel_y = np.transpose(positional_channel_y, [1, 0])
        positional_channels_cache[depth_image_shape_key] = [positional_channel_x, positional_channel_y, ]

      depth_normal_image = np.transpose(depth_normal_image, [2, 0, 1])
      robot_normal_image = np.transpose(robot_normal_image, [2, 0, 1])
      x = [depth_image, depth_image_greater_than_zero, depth_normal_image[0], depth_normal_image[1], depth_normal_image[2],
           robot_image, robot_image_greater_than_zero, robot_normal_image[0], robot_normal_image[1], robot_normal_image[2],
           positional_channel_x, positional_channel_y]
      x = np.transpose(x, [1, 2, 0])
      x = np.array(x)
      
      #rospy.loginfo("  x shape is %s" % str(x.shape))
      
      y = np.array(output_image)
      y = y * output_mask

      #x = x.astype('float32') / 255.0
      #y = y.astype('float32') / 255.0

      batch = (x, y)

      yield batch

    except IOError as e:
      rospy.logerr('cnn_real_image_model_generate: could not load image, error is ' + str(e))
      image_load_ok = False
      pass

    if not image_load_ok:
      scenario_count += 1
      env_count = 0
      view_count = 0
      rospy.loginfo("image_dataset: scenario is now " + str(scenario_num + scenario_count))
      continue

    env_count += 1
    if (env_count >= env_number):
      env_count = 0
      view_count += 1

    if (rospy.is_shutdown()):
      exit()
    pass
  pass

# returns: dataset, x_image_width, x_image_height, y_image_width, y_image_height
def get_image_dataset(scenario_file_name_prefix, image_file_name_prefix, scenario_num, env_number, output_mask):
  y_channels = 1

  generator = get_image_next_kernel(scenario_file_name_prefix,
                                    image_file_name_prefix,
                                    output_mask,
                                    scenario_num, env_number)
  for (x, y) in generator:
    x_image_width = len(x[0])
    x_image_height = len(x)
    y_image_width = len(y[0])
    y_image_height = len(y)
    x_channels = len(x[0][0])
    break

  dataset = tf.data.Dataset.from_generator(lambda: get_image_next_kernel(scenario_file_name_prefix,
                                                                         image_file_name_prefix,
                                                                         output_mask,
                                                                         scenario_num, env_number),
                                           output_types=(tf.float32, tf.float32),
                                           output_shapes=(tf.TensorShape([x_image_height, x_image_width, x_channels]),
                                                          tf.TensorShape([y_image_height, y_image_width, y_channels])))
  return dataset, x_image_width, x_image_height, x_channels, y_image_width, y_image_height
