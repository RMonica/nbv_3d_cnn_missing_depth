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

#include "image_predict.h"

#include "evaluate_view.h"

#include <cv_bridge/cv_bridge.h>

ImagePredict::ImagePredict(ros::NodeHandle & nh): m_nh(nh)
{
  std::string param_string;

  m_nh.param<std::string>(PARAM_NAME_PREDICT_IMAGE_ACTION_NAME, param_string,
                          PARAM_DEFAULT_PREDICT_IMAGE_ACTION_NAME);
  if (param_string.empty())
  {
    ROS_WARN("nbv_3d_cnn_evaluate_view: ImagePredict: empty prediction server name, predictions will not be possible.");
  }
  if (!param_string.empty())
  {
    m_image_predict_action_client.reset(new ImagePredictActionClient(param_string, true));

    ROS_INFO("nbv_3d_cnn_evaluate_view: ImagePredict: waiting for prediction server...");
    m_image_predict_action_client->waitForServer();
    ROS_INFO("nbv_3d_cnn_evaluate_view: ImagePredict: prediction server ok.");
  }
}

bool ImagePredict::Predict(const cv::Mat & depth_image,
                           const cv::Mat & normal_image,
                           const cv::Mat & robot_image,
                           const cv::Mat & robot_normal_image,
                           const cv::Mat & output_mask,
                           cv::Mat & probability_mask)
{
  nbv_3d_cnn_real_image_msgs::ImagePredictGoal goal;

  if (!m_image_predict_action_client)
  {
    ROS_ERROR("nbv_3d_cnn_evaluate_view: ImagePredict: could not predict, server not initialized at startup.");
    return false;
  }

  cv_bridge::CvImagePtr cv_image(new cv_bridge::CvImage);

  cv_image->image = depth_image;
  cv_image->encoding = "32FC1";
  goal.depth_image = *cv_image->toImageMsg();

  cv_image->image = normal_image;
  cv_image->encoding = "32FC3";
  goal.normal_image = *cv_image->toImageMsg();

  cv_image->image = robot_image;
  cv_image->encoding = "32FC1";
  goal.robot_image = *cv_image->toImageMsg();

  cv_image->image = robot_normal_image;
  cv_image->encoding = "32FC3";
  goal.robot_normal_image = *cv_image->toImageMsg();

  const uint64 height = goal.image_height = depth_image.rows;
  const uint64 width = goal.image_width = depth_image.cols;

  goal.output_mask.resize(height * width);
  for (uint64 y = 0; y < height; y++)
    for (uint64 x = 0; x < width; x++)
      goal.output_mask[x + y * width] = output_mask.at<float>(y, x);

  ROS_INFO("nbv_3d_cnn_evaluate_view: ImagePredict: sending goal...");
  m_image_predict_action_client->sendGoal(goal);

  ROS_INFO("nbv_3d_cnn_evaluate_view: ImagePredict: waiting for result...");
  bool finished_before_timeout = m_image_predict_action_client->waitForResult(ros::Duration(30.0));
  if (!finished_before_timeout ||
      m_image_predict_action_client->getState() != actionlib::SimpleClientGoalState::SUCCEEDED)
  {
    ROS_ERROR("nbv_3d_cnn_evaluate_view: ImagePredict: action did not succeed.");
    return false;
  }

  nbv_3d_cnn_real_image_msgs::ImagePredictResult result = *(m_image_predict_action_client->getResult());

  cv_image = cv_bridge::toCvCopy(result.probability_mask);
  probability_mask = cv_image->image;

  if (!probability_mask.data)
  {
    ROS_ERROR("nbv_3d_cnn_evaluate_view: ImagePredict: returned probability mask has no data.");
    return false;
  }

  m_last_prediction_time = result.prediction_time;

  return true;
}
