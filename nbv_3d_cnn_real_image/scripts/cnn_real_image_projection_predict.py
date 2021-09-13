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

from scipy.spatial.transform import Rotation

import cnn_real_image_projection_model

from nbv_3d_cnn_real_image_msgs import msg as nbv_3d_cnn_real_image_msgs

import rospy
import actionlib

from cv_bridge import CvBridge

positional_channels_cache = {}

class PredictAction(object):
  def __init__(self):
    self.last_input_shape = [0, 0]
    self.model = None
    
    self.action_name = rospy.get_param('~action_name', '~predict')

    self.raycasting_downsampling = rospy.get_param('~raycasting_downsampling', 16)

    self.model_filename = rospy.get_param('~model_filename', '')
    
    self.NETWORK_MODE_FULL = 'full'
    self.NETWORK_MODE_ONLY3D = 'only3d'
    self.NETWORK_MODE_UNFROZEN = 'unfrozen'
    self.NETWORK_MODE_NOATTENTION = 'noattention'
    self.network_mode = rospy.get_param('~network_mode', self.NETWORK_MODE_FULL)
    
    rospy.loginfo('cnn_real_image_projection_predict: network mode is \'%s\'.' % str(self.network_mode))
    
    self.action_server = actionlib.SimpleActionServer(self.action_name, nbv_3d_cnn_real_image_msgs.ProjectionPredictAction,
                                                      execute_cb=self.on_predict, auto_start=False)
    self.action_server.start()
    rospy.loginfo('cnn_real_image_projection_predict: action \'%s\' started.' % str(self.action_name))
    
    self.last_model = None
    self.last_model_parameters = None
    pass
    
  def on_predict(self, goal):
    rospy.loginfo("cnn_real_image_projection_predict: on_predict start.")
    result = nbv_3d_cnn_real_image_msgs.ProjectionPredictResult()
    
    default_pose_indices = []
    fx = goal.fx
    fy = goal.fy
    cx = goal.cx
    cy = goal.cy
    image_width = goal.image_width
    image_height = goal.image_height
    image_depth = goal.image_depth
    max_range = goal.max_range
    camera_pose = goal.camera_pose
    pose_rotation = Rotation.from_quat([camera_pose.orientation.x, camera_pose.orientation.y,
                                        camera_pose.orientation.z, camera_pose.orientation.w, ])
    pose_translation = np.array([camera_pose.position.x, camera_pose.position.y, camera_pose.position.z, ], dtype="float32")
    environment_origin = np.array([goal.environment_origin.x, goal.environment_origin.y, goal.environment_origin.z, ],
                                  dtype="float32")
    environment_width = goal.environment_width
    environment_height = goal.environment_height
    environment_depth = goal.environment_depth
    environment_voxel_size = goal.environment_voxel_size

    with_saliency_images = goal.with_saliency_images
    
    input_pose_translation = (pose_translation - environment_origin) / environment_voxel_size
    input_pose_rotation = np.array(pose_rotation.as_matrix(), dtype="float32") / environment_voxel_size

    bridge = CvBridge()
    depth_image = bridge.imgmsg_to_cv2(goal.depth_image, desired_encoding='32FC1')
    depth_normal_image = bridge.imgmsg_to_cv2(goal.normal_image, desired_encoding='32FC3')
    robot_image = bridge.imgmsg_to_cv2(goal.robot_image, desired_encoding='32FC1')
    robot_normal_image = bridge.imgmsg_to_cv2(goal.robot_normal_image, desired_encoding='32FC3')

    x_channels = 12
    input_shape = (image_height, image_width, x_channels)

    depth_image_greater_than_zero = np.array(np.greater(depth_image, 0.0), dtype="float32")
    robot_image_greater_than_zero = np.array(np.greater(robot_image, 0.0), dtype="float32")

    depth_image_shape = depth_image.shape
    depth_image_shape_key = str(depth_image_shape)
    global positional_channels_cache
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
    input_initial_values= [depth_image, depth_image_greater_than_zero, depth_normal_image[0], depth_normal_image[1], depth_normal_image[2],
         robot_image, robot_image_greater_than_zero, robot_normal_image[0], robot_normal_image[1], robot_normal_image[2],
         positional_channel_x, positional_channel_y]
    initial_values_channels = len(input_initial_values)
    input_initial_values = np.transpose(input_initial_values, [1, 2, 0])

    output_mask = goal.output_mask
    output_mask = np.reshape(np.array(output_mask, dtype="float32"), [image_height, image_width, 1])
    
    environment = np.reshape(np.array(goal.ternary_voxelgrid, dtype="float32"),
                             [environment_depth, environment_height, environment_width, 1, ])
    empty_environment = np.array(np.less(environment, -0.5), dtype="float32")
    occupied_environment = np.array(np.greater(environment, 0.5), dtype="float32")
    environment = np.concatenate([empty_environment, occupied_environment, ], axis=3)

    model_parameters = [image_width, image_height, image_depth, environment_width, environment_height, environment_depth,
                        fx, fy, cx, cy, max_range, initial_values_channels]

    raycasting_downsampling = self.raycasting_downsampling
          
    if (self.last_model is None) or (self.last_model_parameters != model_parameters):
      rospy.loginfo("cnn_real_image_projection_predict: parameters changed, generating new model.")
    
      environment_shape = (environment_depth, environment_height, environment_width, 2)
      image_shape = (image_height, image_width, image_depth, 3)

      if self.network_mode == self.NETWORK_MODE_FULL:
        model = cnn_real_image_projection_model.get_real_image_projection_with_saliency_model(environment_shape,
                                                                                image_shape,
                                                                                raycasting_downsampling,
                                                                                initial_values_channels,
                                                                                max_range, fx, fy, cx, cy)
      elif self.network_mode == self.NETWORK_MODE_UNFROZEN:
        model = cnn_real_image_projection_model.get_real_image_projection_with_saliency_model(environment_shape,
                                                                                image_shape,
                                                                                raycasting_downsampling,
                                                                                initial_values_channels,
                                                                                max_range, fx, fy, cx, cy,
                                                                                freeze_autocomplete_layers=False)
      elif self.network_mode == self.NETWORK_MODE_ONLY3D:
        model = cnn_real_image_projection_model.get_real_image_only3d_with_saliency_model(environment_shape,
                                                                            image_shape,
                                                                            raycasting_downsampling,
                                                                            initial_values_channels,
                                                                            max_range, fx, fy, cx, cy)
      elif self.network_mode == self.NETWORK_MODE_NOATTENTION:
        model = cnn_real_image_projection_model.get_real_image_noattention_model(environment_shape,
                                                                                 image_shape,
                                                                                 raycasting_downsampling,
                                                                                 initial_values_channels,
                                                                                 max_range, fx, fy, cx, cy)
      else:
        rospy.logfatal("Unknown network mode: " + self.network_mode)
        self.action_server.set_aborted()
        
      self.last_model = model

      if self.model_filename != "":
        rospy.loginfo("cnn_real_image_projection_predict: loading pre-trained model " + self.model_filename)
        self.last_model.load_weights(self.model_filename)

      self.last_model_parameters = model_parameters
      pass

    prediction_time = rospy.Time.now()
    
    rospy.loginfo("cnn_real_image_projection_predict: predicting.")
    input_tensors = [np.array([environment, ]),
                     np.array([input_pose_rotation, ]),
                     np.array([input_pose_translation, ]),
                     np.array([output_mask, ]),
                     ]
    if self.network_mode != self.NETWORK_MODE_ONLY3D:
      input_tensors.append(np.array([input_initial_values, ]))
    prediction = self.last_model.predict(input_tensors)
        
    projected_environment = prediction[0]
    projected_environment = np.reshape(np.array(projected_environment, dtype="float32"), (image_height, image_width, ))

    prediction_time = rospy.Time.now() - prediction_time
    result.prediction_time = prediction_time.to_sec()
    
    result.image_width = image_width
    result.image_height = image_height
    bridge = CvBridge()
    image_message = bridge.cv2_to_imgmsg(projected_environment, encoding="32FC1")
    result.probability_mask = image_message

    if with_saliency_images and (self.network_mode != self.NETWORK_MODE_NOATTENTION):
      result.saliency_images = []
      saliency_img = prediction[1]
      sal_img = np.array(saliency_img[0], dtype="float32")
      sal_img = np.transpose(sal_img, [3, 0, 1, 2])
      sal_img = np.argmax(sal_img, axis=3).astype(np.float32) * max_range / float(environment_depth)
      for si in sal_img:
        image_message = bridge.cv2_to_imgmsg(si, encoding="32FC1")
        result.saliency_images.append(image_message)
    
    self.action_server.set_succeeded(result)
    rospy.loginfo("cnn_real_image_projection_predict: on_predict end.")
    pass
    
  pass

rospy.init_node('cnn_real_image_projection_predict', anonymous=True)

max_memory_mb = rospy.get_param('~max_memory_mb', 3072)
# limit GPU memory
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(
      gpu, [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=max_memory_mb)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
  except RuntimeError as e:
    print("Exception while limiting GPU memory:")
    print(e)
    exit()

server = PredictAction()
rospy.spin()
