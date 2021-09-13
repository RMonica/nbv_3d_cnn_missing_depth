/*
 * Copyright (c) 2021, Riccardo Monica
 *   RIMLab, Department of Engineering and Architecture, University of Parma, Italy
 *   http://www.rimlab.ce.unipr.it/
 *
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted
 * provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this list of conditions
 * and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice, this list of
 * conditions and the following disclaimer in the documentation and/or other materials provided with
 * the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its contributors may be used to
 * endorse or promote products derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
 * IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
 * OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef AUTOCOMPLETE_PREDICT_H
#define AUTOCOMPLETE_PREDICT_H

// custom
#include <rmonica_voxelgrid_common/voxelgrid.h>
#include <nbv_3d_cnn_msgs/Floats.h>
#include <nbv_3d_cnn_msgs/Predict3dAction.h>

// ROS
#include <ros/ros.h>
#include <actionlib/client/simple_action_client.h>

// STL
#include <vector>
#include <memory>
#include <string>

class AutocompletePredict
{
  public:
  typedef rmonica_voxelgrid_common::Voxelgrid Voxelgrid;

  typedef actionlib::SimpleActionClient<nbv_3d_cnn_msgs::Predict3dAction> Predict3dActionClient;
  typedef std::shared_ptr<Predict3dActionClient> Predict3dActionClientPtr;

  AutocompletePredict(ros::NodeHandle & nh);

  bool Predict3d(const Voxelgrid & empty, const Voxelgrid & occupied, Voxelgrid & autocompleted);

  void DownsampleKinfuVoxelgrid(const Voxelgrid & kinfu_voxelgrid,
                                const float kinfu_voxel_size,
                                const float cnn_voxel_size,
                                const Eigen::Vector3i & kinfu_offset,
                                const Eigen::Vector3i & output_resolution,
                                Voxelgrid & cnn_matrix_empty,
                                Voxelgrid & cnn_matrix_occupied,
                                Voxelgrid & downsampled_voxelgrid);

  private:
  void onRawData(const nbv_3d_cnn_msgs::FloatsConstPtr raw_data);

  ros::NodeHandle & m_nh;

  ros::NodeHandle m_private_nh;
  Predict3dActionClientPtr m_predict3d_action_client;
  ros::Subscriber m_raw_data_subscriber;
  nbv_3d_cnn_msgs::FloatsConstPtr m_raw_data;
  ros::CallbackQueue m_raw_data_callback_queue;
};

#endif // AUTOCOMPLETE_PREDICT_H
