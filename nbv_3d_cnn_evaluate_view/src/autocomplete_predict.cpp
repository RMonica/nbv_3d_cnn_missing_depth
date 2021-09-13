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

#include "autocomplete_predict.h"

#include "evaluate_view.h"

AutocompletePredict::AutocompletePredict(ros::NodeHandle & nh): m_nh(nh), m_private_nh("~")
{
  std::string param_string;

  m_nh.param<std::string>(PARAM_NAME_PREDICT_AUTOCOMPLETE_ACTION_NAME, param_string,
                          PARAM_DEFAULT_PREDICT_AUTOCOMPLETE_ACTION_NAME);
  if (param_string.empty())
  {
    ROS_WARN("nbv_3d_cnn_evaluate_view: AutocompletePredict: empty prediction server name, predictions will not be possible.");
  }
  if (!param_string.empty())
  {
    m_predict3d_action_client.reset(new Predict3dActionClient(param_string, true));

    ROS_INFO("nbv_3d_cnn_evaluate_view: AutocompletePredict: waiting for prediction server...");
    m_predict3d_action_client->waitForServer();
    ROS_INFO("nbv_3d_cnn_evaluate_view: AutocompletePredict: prediction server ok.");
  }

  m_private_nh.setCallbackQueue(&m_raw_data_callback_queue);
  m_raw_data_subscriber = m_private_nh.subscribe(param_string + "raw_data", 1, &AutocompletePredict::onRawData, this);
}

bool AutocompletePredict::Predict3d(const Voxelgrid & empty, const Voxelgrid & occupied, Voxelgrid & autocompleted)
{
  nbv_3d_cnn_msgs::Predict3dGoal goal;

  if (!m_predict3d_action_client)
  {
    ROS_ERROR("nbv_3d_cnn_evaluate_view: Predict3d: could not predict, server not initialized at startup.");
    return false;
  }

  goal.empty = empty.ToFloat32MultiArray();
  goal.frontier = occupied.ToFloat32MultiArray();

  ROS_INFO("nbv_3d_cnn_evaluate_view: Predict3d: sending goal...");
  m_raw_data.reset();
  m_predict3d_action_client->sendGoal(goal);

  ROS_INFO("nbv_3d_cnn_evaluate_view: Predict3d: waiting for result...");
  bool finished_before_timeout = m_predict3d_action_client->waitForResult(ros::Duration(30.0));
  if (!finished_before_timeout ||
      m_predict3d_action_client->getState() != actionlib::SimpleClientGoalState::SUCCEEDED)
  {
    ROS_ERROR("nbv_3d_cnn_evaluate_view: Predict3d: action did not succeed.");
    return false;
  }

  nbv_3d_cnn_msgs::Predict3dResult result = *(m_predict3d_action_client->getResult());
  if (result.scores.data.empty())
  {
    ROS_INFO("nbv_3d_cnn_evaluate_view: Predict3d: waiting for raw data.");
    ros::Rate rate(100);
    while (!m_raw_data)
    {
      m_raw_data_callback_queue.callAvailable(ros::WallDuration());
      rate.sleep();
    }
    result.scores.data = m_raw_data->data;
    m_raw_data.reset();
    ROS_INFO("nbv_3d_cnn_evaluate_view: Predict3d: got raw data.");
  }

  Voxelgrid::Ptr maybe_autocompleted = Voxelgrid::FromFloat32MultiArray(result.scores);
  if (!maybe_autocompleted)
  {
    ROS_ERROR("nbv_3d_cnn_evaluate_view: Predict3d: could not convert result to voxelgrid.");
    return false;
  }
  autocompleted = *maybe_autocompleted;
  return true;
}

void AutocompletePredict::DownsampleKinfuVoxelgrid(const Voxelgrid & kinfu_voxelgrid,
                                                   const float kinfu_voxel_size,
                                                   const float cnn_voxel_size,
                                                   const Eigen::Vector3i & kinfu_offset,
                                                   const Eigen::Vector3i & output_resolution,
                                                   Voxelgrid & cnn_matrix_empty,
                                                   Voxelgrid & cnn_matrix_occupied,
                                                   Voxelgrid & downsampled_voxelgrid)
{
  cnn_matrix_empty = Voxelgrid(output_resolution);
  cnn_matrix_empty.Fill(1.0);
  cnn_matrix_occupied = Voxelgrid(output_resolution);
  cnn_matrix_occupied.Fill(0.0);
  downsampled_voxelgrid = Voxelgrid(output_resolution);
  downsampled_voxelgrid.Fill(-1.0);

  {
    const Eigen::Vector3i kinfu_voxelgrid_size = kinfu_voxelgrid.GetSize();

    Eigen::Vector3i xyz;
    for (xyz.z() = 0; xyz.z() < kinfu_voxelgrid_size.z(); xyz.z()++)
      for (xyz.y() = 0; xyz.y() < kinfu_voxelgrid_size.y(); xyz.y()++)
        for (xyz.x() = 0; xyz.x() < kinfu_voxelgrid_size.x(); xyz.x()++)
        {
          const Eigen::Vector3i cnn_matrix_xyz = (((xyz - kinfu_offset).cast<float>() + Eigen::Vector3f::Ones() * 0.5f) *
                                                  kinfu_voxel_size /
                                                  cnn_voxel_size
                                                  ).array().floor().cast<int>();

          if ((cnn_matrix_xyz.array() < 0).any() ||
              (cnn_matrix_xyz.array() >= output_resolution.array()).any())
            continue;

          const float prev_cnn_matrix_empty = cnn_matrix_empty.at(cnn_matrix_xyz);

          // priority: unknown > occupied > empty

          const float v = kinfu_voxelgrid.at(xyz);
          // if (v < -0.5f) empty do nothing

          if (v > 0.5f && prev_cnn_matrix_empty) // occupied overrides empty
          {
            cnn_matrix_empty.at(cnn_matrix_xyz) = 0.0f;
            cnn_matrix_occupied.at(cnn_matrix_xyz) = 1.0f;
            downsampled_voxelgrid.at(cnn_matrix_xyz) = 1.0f;
          }

          if (v > -0.5f && v < 0.5f) // unknown overrides all
          {
            cnn_matrix_empty.at(cnn_matrix_xyz) = 0.0f;
            cnn_matrix_occupied.at(cnn_matrix_xyz) = 0.0f;
            downsampled_voxelgrid.at(cnn_matrix_xyz) = 0.0f;
          }
        }
  }
}

void AutocompletePredict::onRawData(const nbv_3d_cnn_msgs::FloatsConstPtr raw_data)
{
  m_raw_data = raw_data;
  ROS_INFO("AutocompleteIGainNBVAdapter: got raw data.");
}
