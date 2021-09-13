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

#ifndef EVALUATE_VIEW_FILTER_H
#define EVALUATE_VIEW_FILTER_H

#include <stdint.h>
#include <cmath>
#include <string>

#include "evaluate_view_opencl.h"

class EvaluateViewFilter
{
  public:
  typedef uint64_t uint64;
  typedef int64_t int64;
  typedef uint8_t uint8;
  typedef std::vector<bool> BoolVector;

  struct ReachingResult
  {
    cv::Mat depth;  // if the ray reaches this depth
    cv::Mat status; // then this is the status
    cv::Mat normals;

    enum
    {
      UNKNOWN  = 0, // the status is unknown
      OCCUPIED = 1, // the sensor sees an occupied pixel
      LOST     = 2, // the sensor sees nothing (ray is lost to infinity or below min range)
    };

    ReachingResult(const uint64 width, const uint64 height)
    {
      depth = cv::Mat(height, width, CV_32FC1, 0.0f);
      normals = cv::Mat(height, width, CV_32FC3, cv::Vec3f(0.0f, 0.0f, 0.0f));
      status = cv::Mat(height, width, CV_8UC1, UNKNOWN);
    }

    ReachingResult(const ReachingResult & other)
    {
      depth = other.depth.clone();
      status = other.status.clone();
      normals = other.normals.clone();
    }

    ReachingResult & operator=(const ReachingResult & other)
    {
      depth = other.depth.clone();
      status = other.status.clone();
      normals = other.normals.clone();

      return *this;
    }
  };
  typedef std::shared_ptr<ReachingResult> ReachingResultPtr;

  EvaluateViewFilter(ros::NodeHandle & nh);

  ReachingResult Filter(const EvaluateViewOpenCL::RaycastResult & raycast_result,
                        const float voxel_size,
                        const Eigen::Vector2f &camera_center,
                        const Eigen::Vector2f &camera_focal,
                        const float min_range,
                        const cv::Mat & robot_depth_image,
                        const cv::Mat & robot_normal_image);

  ReachingResult FilterAddRobot(const ReachingResult & image,
                                const cv::Mat & robot_depth_image,
                                const cv::Mat & robot_normal_image);

  cv::Mat GenerateFilterCircleRectangleMask(const uint64 width, const uint64 height) const;

  ReachingResult FilterCircleRectangle(const ReachingResult & image);
  ReachingResult FilterDiscontinuity(const ReachingResult & image);
  ReachingResult FilterNormalInclination(const ReachingResult & image,
                                  const Eigen::Vector2f &camera_center,
                                  const Eigen::Vector2f &camera_focal);
  ReachingResult PixelShift(const ReachingResult &image);
  ReachingResult ShadowRemoval(const Eigen::Vector2f &camera_center,
                               const Eigen::Vector2f &camera_focal,
                               const float shadow_removal_emitter_distance,
                               const ReachingResult & intensity_in);

  ReachingResult FilterMinRange(const ReachingResult &image, const float min_range);

  ReachingResult FilterOpening(const ReachingResult &image, const uint64 opening_size);
  ReachingResult FilterErosion(const ReachingResult &image, const uint64 erosion_size);
  ReachingResult FilterErosionByDistance(const ReachingResult &image,
                                         const uint64 erosion_size,
                                         const float depth_scale);

  cv::Mat GetLastNormals() {return m_last_normals.clone(); }

  private:
  cv::Mat m_last_normals;

  float m_center_x;
  float m_center_y;
  float m_circle_radius;

  float m_rectangle_width;
  float m_rectangle_height;
  float m_rectangle_x;
  float m_rectangle_y;

  float m_normal_max_angle;

  uint64 m_opening_kernel_size;
  uint64 m_erosion_kernel_size;
  float m_erosion_depth_scale;

  float m_discontinuity_filter_th;
  uint64 m_discontinuity_filter_window;

  int64 m_pixel_shift_x;
  int64 m_pixel_shift_y;

  float m_shadow_removal_emitter_distance_left;
  float m_shadow_removal_emitter_distance_right;

  ros::NodeHandle & m_nh;
};

#endif // EVALUATE_VIEW_FILTER_H
