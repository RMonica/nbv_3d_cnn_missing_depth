#!/usr/bin/python3

import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
import re
import math

import os

import rospy

def load_voxelgrid(infilename):
  ifile = open(infilename + ".binvoxelgrid", "r")
  metadata = np.fromfile(ifile, dtype=np.uint32, count=5)
  
  if (int(metadata[0]).to_bytes(4, 'little') != bytes("VXGR", 'utf-8')):
    rospy.logfatal("load_voxelgrid: VXGR expected, %s found." % str(bytes([metadata[0], ])))
    exit()
    
  version = metadata[1]
  if (version > 1):
    rospy.logfatal("load_voxelgrid: unsupported version %u." % version)
    exit()
  
  width = metadata[2]
  height = metadata[3]
  depth = metadata[4]

  voxelgrid = np.fromfile(ifile, dtype=np.float32, count=width*height*depth)
  voxelgrid = np.reshape(voxelgrid, [depth, height, width])

  ifile.close()

  return voxelgrid

def save_voxelgrid(outfilename, npmatrix):
  ofile = open(outfilename + ".binvoxelgrid", "wb")
  ofile.write(bytes("VXGR", 'utf-8'))

  version = 1
  width = npmatrix.shape[2]
  height = npmatrix.shape[1]
  depth = npmatrix.shape[0]
  metadata = np.asarray([version, width, height, depth], dtype=np.uint32)
  ofile.write(metadata.tobytes())

  npmatrix = npmatrix.astype('float32')
  ofile.write(npmatrix.tobytes())

  ofile.close()
  pass

def save_voxelgrid_nchannels(outfilename, npmatrix):
  trmatrix = np.transpose(npmatrix, [3, 0, 1, 2])
  for i in range(0, len(trmatrix)):
    save_voxelgrid(outfilename + str(i), trmatrix[i])
  pass

def load_voxelgrid4(infilename):
  result = []
  for i in range(0, 4):
    result.append(load_voxelgrid(infilename + str(i)))
  result = np.transpose(result, [1, 2, 3, 0])
  return result

def get_scene_3d_dataset_next_kernel(source_file_name_prefix, x_empty_prefix,
                                     x_frontier_prefix, y_prefix, y_channels, enable_augmentation,
                                     start, end):
  image_load_ok = True
  counter = start
  augmentation_counter = 0

  rotations = [[0, 0, 0], ]
  if ('rotation4' in enable_augmentation):
    rotations = []
    for c in range(0, 4):
      rotations.append([c, 0, 0])

  while (image_load_ok):
    sub_file_suffix = ""
    sub_file_gt_suffix = ""

    empty_filename = source_file_name_prefix + str(counter) + x_empty_prefix + sub_file_suffix
    frontier_filename = source_file_name_prefix + str(counter) + x_frontier_prefix + sub_file_suffix
    gt_filename = source_file_name_prefix + str(counter) + y_prefix + sub_file_suffix + sub_file_gt_suffix

    try:
      for rotation in rotations:
        #rospy.loginfo("nbv_3d_cnn: loading empty '%s'" % frontier_filename)
        empty = load_voxelgrid(empty_filename)
        #rospy.loginfo("nbv_3d_cnn: loading frontier '%s'" % frontier_filename)
        frontier = load_voxelgrid(frontier_filename)
        #rospy.loginfo("nbv_3d_cnn: loading gt '%s'" % gt_filename)
        gt = load_voxelgrid(gt_filename)

        empty = np.rot90(np.array(empty), k=rotation[2], axes=(0, 1))
        frontier = np.rot90(np.array(frontier), k=rotation[2], axes=(0, 1))
        gt = np.rot90(np.array(gt), k=rotation[2], axes=(0, 1))

        empty = np.rot90(np.array(empty), k=rotation[1], axes=(0, 2))
        frontier = np.rot90(np.array(frontier), k=rotation[1], axes=(0, 2))
        gt = np.rot90(np.array(gt), k=rotation[1], axes=(0, 2))

        empty = np.rot90(np.array(empty), k=rotation[0], axes=(1, 2))
        frontier = np.rot90(np.array(frontier), k=rotation[0], axes=(1, 2))
        gt = np.rot90(np.array(gt), k=rotation[0], axes=(1, 2))

        x = [empty, frontier]
        x = np.transpose(x, [1, 2, 3, 0])
        x = np.array(x)

        gt = np.reshape(gt, [len(gt), len(gt[0]), len(gt[0][0]) // y_channels, y_channels])
        y = np.array(gt)

        x = x.astype('float32')
        y = y.astype('float32')

        batch = ((x, ), (y, ))

        yield batch
      pass

    except IOError as e:
      rospy.logerr('nbv_3d_cnn: could not load image, error is ' + str(e))
      image_load_ok = False
      pass

    counter += 1

    if (counter >= end):
      return

    if (rospy.is_shutdown()):
      exit()
    pass
  pass

# returns: dataset, x_image_width, x_image_height, x_image_depth, y_image_width, y_image_height, y_image_depth,
def get_scene_3d_dataset(source_file_name_prefix, start, end, enable_augmentation):
  empty_prefix = '_empty';
  output_prefix = '_environment'
  frontier_prefix = '_occupied'
  y_channels = 1

  generator = get_scene_3d_dataset_next_kernel(source_file_name_prefix,
                                               empty_prefix,
                                               frontier_prefix,
                                               output_prefix,
                                               y_channels,
                                               enable_augmentation,
                                               start, end)

  for ((x, ), (y, )) in generator:
    x_image_width = len(x[0][0])
    x_image_height = len(x[0])
    x_image_depth = len(x)
    y_image_width = len(y[0][0])
    y_image_height = len(y[0])
    y_image_depth = len(y)
    break

  dataset = tf.data.Dataset.from_generator(lambda: get_scene_3d_dataset_next_kernel(source_file_name_prefix,
                                                                                    empty_prefix,
                                                                                    frontier_prefix,
                                                                                    output_prefix,
                                                                                    y_channels,
                                                                                    enable_augmentation,
                                                                                    start, end),
                                           output_types=(tf.float32, tf.float32),
                                           output_shapes=(tf.TensorShape([1, x_image_depth, x_image_height, x_image_width, 2]),
                                                          tf.TensorShape([1, y_image_depth, y_image_height,
                                                                          y_image_width, y_channels])))
  return dataset, x_image_width, x_image_height, x_image_depth, y_image_width, y_image_height, y_image_depth

