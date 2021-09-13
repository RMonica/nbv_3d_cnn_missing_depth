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

#ifndef IMAGE_PREDICT_H
#define IMAGE_PREDICT_H

// custom
#include <rmonica_voxelgrid_common/voxelgrid.h>

#include <nbv_3d_cnn_real_image_msgs/ImagePredictAction.h>

// ROS
#include <ros/ros.h>
#include <actionlib/client/simple_action_client.h>

// STL
#include <vector>
#include <memory>
#include <string>
#include <stdint.h>

class ImagePredict
{
  public:
  typedef actionlib::SimpleActionClient<nbv_3d_cnn_real_image_msgs::ImagePredictAction> ImagePredictActionClient;
  typedef std::shared_ptr<ImagePredictActionClient> ImagePredictActionClientPtr;

  typedef uint64_t uint64;

  ImagePredict(ros::NodeHandle & nh);

  bool Predict(const cv::Mat & depth_image,
               const cv::Mat & normal_image,
               const cv::Mat & robot_image,
               const cv::Mat & robot_normal_image,
               const cv::Mat & output_mask,
               cv::Mat & probability_mask);

  double GetLastPredictionTime() const {return m_last_prediction_time; }

  private:

  ros::NodeHandle & m_nh;

  double m_last_prediction_time;

  ImagePredictActionClientPtr m_image_predict_action_client;
};

#endif // IMAGE_PREDICT_H
