#!/usr/bin/env python3

import tensorflow as tf

import numpy as np
import math
import sys

from cv_bridge import CvBridge

import PIL

import cnn_real_image_model

from nbv_3d_cnn_real_image_msgs import msg as nbv_3d_cnn_real_image_msgs

import rospy
import actionlib

def greater_than_zero(v):
  if v > 0.0:
    return 1.0
  else:
    return 0.0
vectorized_greater_than_zero = np.vectorize(greater_than_zero)

class PredictAction(object):
  def __init__(self):
    self.last_input_shape = [0, 0]
    self.model = None
    
    self.action_name = rospy.get_param('~action_name', '~predict')

    self.model_filename = rospy.get_param('~model_filename', '')

    self.max_range = rospy.get_param('~max_range', '')
    
    self.action_server = actionlib.SimpleActionServer(self.action_name, nbv_3d_cnn_real_image_msgs.ImagePredictAction,
                                                      execute_cb=self.on_predict, auto_start=False)
    self.action_server.start()
    rospy.loginfo('cnn_real_image_predict: action \'%s\' started.' % self.action_name)
    
    self.last_model = None
    self.last_model_parameters = None
    pass
    
  def on_predict(self, goal):
    rospy.loginfo("cnn_real_image_projection_predict: on_predict start.")
    result = nbv_3d_cnn_real_image_msgs.ImagePredictResult()
    
    image_width = goal.image_width
    image_height = goal.image_height

    bridge = CvBridge()
    depth_image = bridge.imgmsg_to_cv2(goal.depth_image, desired_encoding='32FC1')
    depth_normal_image = bridge.imgmsg_to_cv2(goal.normal_image, desired_encoding='32FC3')
    robot_image = bridge.imgmsg_to_cv2(goal.robot_image, desired_encoding='32FC1')
    robot_normal_image = bridge.imgmsg_to_cv2(goal.robot_normal_image, desired_encoding='32FC3')

    max_range = self.max_range

    x_channels = 12
    input_shape = (image_height, image_width, x_channels)

    depth_image_greater_than_zero = vectorized_greater_than_zero(depth_image)
    robot_image_greater_than_zero = vectorized_greater_than_zero(robot_image)

    positional_channel_x = [list(range(0, input_shape[1])) for i in range(0, input_shape[0])]
    positional_channel_x = np.array(positional_channel_x).astype('float32') / float(input_shape[1])
    positional_channel_y = [list(range(0, input_shape[0])) for i in range(0, input_shape[1])]
    positional_channel_y = np.array(positional_channel_y).astype('float32') / float(input_shape[0])
    positional_channel_y = np.transpose(positional_channel_y, [1, 0])

    depth_normal_image = np.transpose(depth_normal_image, [2, 0, 1])
    robot_normal_image = np.transpose(robot_normal_image, [2, 0, 1])
    x = [depth_image, depth_image_greater_than_zero, depth_normal_image[0], depth_normal_image[1], depth_normal_image[2],
         robot_image, robot_image_greater_than_zero, robot_normal_image[0], robot_normal_image[1], robot_normal_image[2],
         positional_channel_x, positional_channel_y]
    x = np.transpose(x, [1, 2, 0])
    x = np.array([x, ])

    output_mask = goal.output_mask
    output_mask = np.reshape(np.array(output_mask, dtype="float32"), [image_height, image_width, 1])

    model_parameters = [image_width, image_height, ]
          
    if (self.last_model is None) or (self.last_model_parameters != model_parameters):
      rospy.loginfo("cnn_real_image_predict: parameters changed, generating new model.")
    
      self.last_model = cnn_real_image_model.get_real_image_model(input_shape, output_mask, max_range)
      self.last_model.load_weights(self.model_filename)

      self.last_model_parameters = model_parameters
      pass

    prediction_time = rospy.Time.now()
    
    rospy.loginfo("cnn_real_image_predict: predicting.")
    predicted_image = self.last_model.predict(x)

    prediction_time = rospy.Time.now() - prediction_time
    result.prediction_time = prediction_time.to_sec()

    predicted_image = np.array(predicted_image, dtype='float32')
    predicted_image = predicted_image[0]
    predicted_image = np.clip(predicted_image, 0.0, 1.0)
    
    result.image_width = image_width
    result.image_height = image_height
    result.probability_mask = bridge.cv2_to_imgmsg(predicted_image, encoding="32FC1")
    
    self.action_server.set_succeeded(result)
    rospy.loginfo("cnn_real_image_predict: on_predict end.")
    pass
    
  pass

rospy.init_node('cnn_real_image_predict', anonymous=True)

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
