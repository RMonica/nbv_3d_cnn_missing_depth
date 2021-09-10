#!/usr/bin/env python3

import tensorflow as tf
import tensorflow_probability as tfp

import numpy as np
import math
import sys

import cnn_real_image_projection_model

default_pose_indices = []
SCALE = 20
MAX_RANGE = 1.6
RANGE_STEP = 0.01
VOXEL_SIZE = 0.011
fx = 517.177 / SCALE
fy = 518.680 / SCALE
cx = 313.199 / SCALE
cy = 238.733 / SCALE
width = 640 // SCALE
height = 480 // SCALE
depth = int(MAX_RANGE / RANGE_STEP) // SCALE
for yi in range(0, height):
  default_pose_indices_y = []
  for xi in range(0, width):
    default_pose_indices_x = []
    for zi in range(0, depth):
      z = zi * RANGE_STEP * SCALE
      x = (xi - cx) / fx * z
      y = (yi - cx) / fy * z
      default_pose_indices_x.append([x, y, z, ])
    default_pose_indices_y.append(default_pose_indices_x)
  default_pose_indices.append(default_pose_indices_y)
default_pose_indices = np.array([default_pose_indices, ], dtype="float64")
  
test_environment = np.array([[[
  [10.0, 9.0, 8.0, 7.0],
  [ 9.0, 8.0, 7.0, 6.0],
  [ 8.0, 7.0, 6.0, 5.0],
  [ 7.0, 6.0, 5.0, 4.0],
  ],
  [
  [ 9.0, 8.0, 7.0, 6.0],
  [ 8.0, 7.0, 6.0, 5.0],
  [ 7.0, 6.0, 5.0, 4.0],
  [ 6.0, 5.0, 4.0, 3.0],
  ],
  [
  [ 8.0, 7.0, 6.0, 5.0],
  [ 7.0, 6.0, 5.0, 4.0],
  [ 6.0, 5.0, 4.0, 3.0],
  [ 5.0, 4.0, 3.0, 2.0],
  ],
  [
  [ 7.0, 6.0, 5.0, 4.0],
  [ 6.0, 5.0, 4.0, 3.0],
  [ 5.0, 4.0, 3.0, 2.0],
  [ 4.0, 3.0, 2.0, 1.0],
  ],],])
  
pose_rotation = np.array([[
  [ 1.0, 0.0, 0.0,],
  [ 0.0, 1.0, 0.0,],
  [ 0.0, 0.0, 1.0,],
  ],])
pose_translation = np.array([[0.0, ], [0.0, ], [-1.0, ],])

environment_shape=(4, 4, 4, 1)
image_shape=(height, width, depth, 3)

pose_indices = np.array(default_pose_indices)


print("test environment shape is " + str(test_environment.shape))
print("pose indices shape is " + str(pose_indices.shape))
  
model = cnn_real_image_projection_model.get_real_image_projection_model(environment_shape, image_shape)
projected_environment = model.predict([test_environment, pose_indices])
print(projected_environment)
