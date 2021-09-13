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

environment_cache = {}
environment_metadata_cache = {}
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
    exit(1)
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
    exit(1)
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
    exit(1)
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

def load_voxelgrid(infilename):
  ifile = open(infilename, "rb")
  metadata = np.fromfile(ifile, dtype=np.uint32, count=5)
  
  if (int(metadata[0]).to_bytes(4, 'little') != bytes("VXGT", 'utf-8')):
    rospy.logfatal("load_voxelgrid: VXGR expected, %s found." % str(int(metadata[0]).to_bytes(4, 'little')))
    exit()
    
  version = metadata[1]
  if (version > 1):
    rospy.logfatal("load_voxelgrid: unsupported version %u." % version)
    exit()
  
  width = metadata[2]
  height = metadata[3]
  depth = metadata[4]

  voxelgrid = np.fromfile(ifile, dtype=np.int8, count=width*height*depth)
  voxelgrid = np.reshape(voxelgrid, [depth, height, width])
  voxelgrid = np.array(voxelgrid, dtype="float32")

  ifile.close()

  return voxelgrid
  
def load_pose_rotation_translation(filename):
  pose = np.loadtxt(filename, dtype="float32")
  rotation = pose[:3, :3]
  translation = pose[:3, 3]
  return rotation, translation
  
def load_metadata(filename):
  ifile = open(filename, "r")
  lines = ifile.readlines()
  for l in lines:
    splitted = l.split()
    if splitted[0] == "BBOX_MIN":
      bbox_min = np.fromstring(splitted[1] + " " + splitted[2] + " " + splitted[3], dtype="float32", sep=' ')
    if splitted[0] == "BBOX_MAX":
      bbox_max = np.fromstring(splitted[1] + " " + splitted[2] + " " + splitted[3], dtype="float32", sep=' ')
    if splitted[0] == "VOXEL_SIZE":
      voxel_size = np.fromstring(splitted[1], dtype="float32",sep=' ')
  
  ifile.close()
  return {'bbox_min': bbox_min, 'bbox_max': bbox_max, 'voxel_size': voxel_size[0]}

def get_image_next_kernel(scenario_file_name_prefix,
                          environment_file_name_prefix,
                          image_file_name_prefix,
                          pose_file_name_prefix,
                          output_mask,
                          scenario_num, env_number,
                          zero_initial_values):
  scenario_count = 0
  view_count = 0
  env_count = 0
  
  global environment_cache
  global environment_metadata_cache
  global positional_channels_cache
  
  while (scenario_count < 1):
    scenario_prefix = scenario_file_name_prefix + str(scenario_num)
    depth_filename = scenario_prefix + image_file_name_prefix + "raw_depth_" + str(env_count) + "_" + str(view_count) + ".png"
    depth_normal_filename = scenario_prefix + image_file_name_prefix + "raw_normal_" + str(env_count) + "_" + str(view_count) + ".png"
    robot_filename = scenario_prefix + image_file_name_prefix + "raw_robot_" + str(env_count) + "_" + str(view_count) + ".png"
    robot_normal_filename = scenario_prefix + image_file_name_prefix + "raw_robot_normal_" + str(env_count) + "_" + str(view_count) + ".png"
    pose_filename = scenario_prefix + pose_file_name_prefix + "pose_" + str(view_count) + ".matrix"
    output_filename = scenario_prefix + image_file_name_prefix + "gt_" + str(view_count) + ".png"
    
    environment_filename = scenario_prefix + environment_file_name_prefix + 'cropped_' + str(env_count) + ".voxelgrid"
    environment_metadata_filename = scenario_prefix + environment_file_name_prefix + 'cropped_' + str(env_count) + ".metadata"

    image_load_ok = True
    try:
          
      # cached environment
      if environment_filename in environment_cache:
        input_environment = environment_cache[environment_filename]
        environment_metadata = environment_metadata_cache[environment_filename]
      else:
        environment = load_voxelgrid(environment_filename)
        environment_occupied = np.array(np.greater(environment, 0.5), dtype="float32")
        environment_empty = np.array(np.less(environment, -0.5), dtype="float32")
        input_environment = [environment_empty, environment_occupied]
        input_environment = np.transpose(input_environment, [1, 2, 3, 0])
        input_environment = np.array(input_environment, dtype='float32')
        environment_metadata = load_metadata(environment_metadata_filename)
        environment_cache[environment_filename] = input_environment
        environment_metadata_cache[environment_filename] = environment_metadata
   
      rotation, translation = load_pose_rotation_translation(pose_filename)
    
      input_pose_translation = (translation - environment_metadata['bbox_min']) / environment_metadata['voxel_size']
      input_pose_rotation = rotation / environment_metadata['voxel_size']
    
      depth_image = load_image_16bit_cached(depth_filename)
      depth_normal_image = load_image_color_cached(depth_normal_filename)
      robot_image = load_image_color_cached(robot_filename)
      robot_normal_image = load_image_color_cached(robot_normal_filename)
      
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
      initial_values = [depth_image, depth_image_greater_than_zero, depth_normal_image[0], depth_normal_image[1], depth_normal_image[2],
                        robot_image, robot_image_greater_than_zero, robot_normal_image[0], robot_normal_image[1], robot_normal_image[2],
                        positional_channel_x, positional_channel_y]
      initial_values = np.transpose(initial_values, [1, 2, 0])
      initial_values = np.array(initial_values, dtype="float32")
      if zero_initial_values:
        initial_values.fill(0.0)
          
      output_image = load_image_8bit_cached(output_filename)
      output_image = [output_image, ]
      output_image = np.transpose(output_image, [1, 2, 0])
      
      y = np.array(output_image, dtype="float32")
      y = y * output_mask
      
      x = {
        'input_environment': input_environment, 
        'input_pose_rotation': input_pose_rotation, 
        'input_pose_translation': input_pose_translation, 
        'input_output_mask': output_mask, 
        'input_initial_values': initial_values
      }

      batch = (x, y)

      yield batch
      
      if (rospy.is_shutdown()):
        exit()

    except IOError as e:
      rospy.logwarn('image_and_scene_3d_dataset: could not load image, error is ' + str(e))
      image_load_ok = False
      pass
      
    if not image_load_ok:
      scenario_count += 1
      env_count = 0
      view_count = 0
      rospy.loginfo("image_and_scene_3d_dataset: scenario is now " + str(scenario_num + scenario_count))
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
def get_image_dataset(scenario_file_name_prefix,
                      environment_file_name_prefix,
                      image_file_name_prefix,
                      pose_file_name_prefix, scenario_num, env_number, 
                      output_mask,
                      zero_initial_values=False):
  y_channels = 1
  environment_channels = 2
  
  # environment
  # input_pose_rotation
  # input_pose_translation
  # output_mask
  # input_initial_values

  generator = get_image_next_kernel(scenario_file_name_prefix,
                                    environment_file_name_prefix,
                                    image_file_name_prefix,
                                    pose_file_name_prefix,
                                    output_mask,
                                    scenario_num, env_number,
                                    zero_initial_values)
                                    
  at_least_one = False
  for (x, y) in generator:
    x_image_width = len(x['input_initial_values'][0])
    x_image_height = len(x['input_initial_values'])
    x_channels = len(x['input_initial_values'][0][0])
    scene_3d_width = len(x['input_environment'][0][0])
    scene_3d_height = len(x['input_environment'][0])
    scene_3d_depth = len(x['input_environment'])
    at_least_one = True
    
    #print("input_environment: " + str(x['input_environment'].shape))
    #print("input_pose_rotation: " + str(x['input_pose_rotation'].shape))
    #print("input_pose_translation: " + str(x['input_pose_translation'].shape))
    #print("input_output_mask: " + str(x['input_output_mask'].shape))
    #print("input_initial_values: " + str(x['input_initial_values'].shape))
    break
    
  if not at_least_one:
    rospy.logfatal("image_and_scene_3d_dataset: no image could be loaded!")
    exit()

  dataset = tf.data.Dataset.from_generator(lambda: get_image_next_kernel(scenario_file_name_prefix,
                                                                         environment_file_name_prefix,
                                                                         image_file_name_prefix,
                                                                         pose_file_name_prefix,
                                                                         output_mask,
                                                                         scenario_num, env_number,
                                                                         zero_initial_values),
                                           output_types=({'input_environment': tf.float32, 
                                                          'input_pose_rotation': tf.float32, 
                                                          'input_pose_translation': tf.float32, 
                                                          'input_output_mask': tf.float32, 
                                                          'input_initial_values': tf.float32}, tf.float32)
                                           )
  
  image_metadata = {'image_width': x_image_width, 
                    'image_height': x_image_height, 
                    'image_channels': x_channels,
                    }
  scene_3d_metadata = {'scene_3d_width': scene_3d_width,
                       'scene_3d_height': scene_3d_height,
                       'scene_3d_depth': scene_3d_depth, 
                       }

  return dataset, image_metadata, scene_3d_metadata
