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

#include "evaluate_view_filter.h"

#include "opencv2/imgproc.hpp"

#include "evaluate_view.h"

template <typename T>
static T SQR(const T & t) {return t * t; }

EvaluateViewFilter::EvaluateViewFilter(ros::NodeHandle & nh):
  m_nh(nh)
{
  double param_double;
  std::string param_string;
  int param_int;

  nh.param<double>(PARAM_NAME_FILTER_CIRCLE_RADIUS, param_double, PARAM_DEFAULT_FILTER_CIRCLE_RADIUS);
  m_circle_radius = param_double;

  nh.param<std::string>(PARAM_NAME_FILTER_CIRCLE_CENTER, param_string, PARAM_DEFAULT_FILTER_CIRCLE_CENTER);
  std::istringstream istr(param_string);
  istr >> m_center_x >> m_center_y;
  if (!istr)
  {
    ROS_ERROR("EvaluateViewFilter: circle_filter: could not parse circle center '%s'", param_string.c_str());
    std::istringstream istr2(PARAM_DEFAULT_FILTER_CIRCLE_CENTER);
    istr >> m_center_x >> m_center_y;
  }

  m_nh.param<double>(PARAM_NAME_FILTER_RECTANGLE_WIDTH, param_double, PARAM_DEFAULT_FILTER_RECTANGLE_WIDTH);
  m_rectangle_width = param_double;

  m_nh.param<double>(PARAM_NAME_FILTER_RECTANGLE_HEIGHT, param_double, PARAM_DEFAULT_FILTER_RECTANGLE_HEIGHT);
  m_rectangle_height = param_double;

  m_nh.param<double>(PARAM_NAME_FILTER_RECTANGLE_X, param_double, PARAM_DEFAULT_FILTER_RECTANGLE_X);
  m_rectangle_x = param_double;

  m_nh.param<double>(PARAM_NAME_FILTER_RECTANGLE_Y, param_double, PARAM_DEFAULT_FILTER_RECTANGLE_Y);
  m_rectangle_y = param_double;

  m_nh.param<double>(PARAM_NAME_FILTER_DISCONTINUITY_TH, param_double, PARAM_DEFAULT_FILTER_DISCONTINUITY_TH);
  m_discontinuity_filter_th = param_double;

  m_nh.param<int>(PARAM_NAME_FILTER_DISCONTINUITY_WIN, param_int, PARAM_DEFAULT_FILTER_DISCONTINUITY_WIN);
  m_discontinuity_filter_window = param_int;

  m_nh.param<int>(PARAM_NAME_FILTER_PIXEL_SHIFT_X, param_int, PARAM_DEFAULT_FILTER_PIXEL_SHIFT_X);
  m_pixel_shift_x = param_int;

  m_nh.param<int>(PARAM_NAME_FILTER_PIXEL_SHIFT_Y, param_int, PARAM_DEFAULT_FILTER_PIXEL_SHIFT_Y);
  m_pixel_shift_y = param_int;

  m_nh.param<int>(PARAM_NAME_FILTER_OPENING_SIZE, param_int, PARAM_DEFAULT_FILTER_OPENING_SIZE);
  m_opening_kernel_size = param_int;

  m_nh.param<int>(PARAM_NAME_FILTER_EROSION_SIZE, param_int, PARAM_DEFAULT_FILTER_EROSION_SIZE);
  m_erosion_kernel_size = param_int;

  m_nh.param<double>(PARAM_NAME_FILTER_EROSION_DEPTH_SCALE, param_double, PARAM_DEFAULT_FILTER_EROSION_DEPTH_SCALE);
  m_erosion_depth_scale = param_double;

  m_nh.param<double>(PARAM_NAME_FILTER_NORMAL_MAX_ANGLE, param_double, PARAM_DEFAULT_FILTER_NORMAL_MAX_ANGLE);
  m_normal_max_angle = param_double / 180.0 * M_PI;

  m_nh.param<double>(PARAM_NAME_SHADOW_REMOVAL_EMITTER_DISTANCE_RIGHT, param_double, PARAM_DEFAULT_SHADOW_REMOVAL_EMITTER_DISTANCE_RIGHT);
  m_shadow_removal_emitter_distance_right = param_double;

  m_nh.param<double>(PARAM_NAME_SHADOW_REMOVAL_EMITTER_DISTANCE_LEFT, param_double, PARAM_DEFAULT_SHADOW_REMOVAL_EMITTER_DISTANCE_LEFT);
  m_shadow_removal_emitter_distance_left = param_double;
}

EvaluateViewFilter::ReachingResult EvaluateViewFilter::ShadowRemoval(const Eigen::Vector2f & camera_center,
                                                                     const Eigen::Vector2f & camera_focal,
                                                                     const float shadow_removal_emitter_distance,
                                                                     const ReachingResult & image)
{
  ReachingResult result = image;

  if (!shadow_removal_emitter_distance)
    return result;

  const float fx = camera_focal.x();
  const float fy = camera_focal.y();
  const float cx = camera_center.x();
  const float cy = camera_center.y();

  const float emitter_distance = shadow_removal_emitter_distance;

  const uint64 width = result.depth.cols;
  const uint64 height = result.depth.rows;

  int xstart = width - 1;
  int xend = -1;
  int xinc = -1;

  if (emitter_distance < 0.0f)
  {
    xstart = 0;
    xend = width;
    xinc = 1;
  }

  const float xincf = -xinc;

  for (int64 yi = 0; yi < height; yi++)
  {
    float prev_ratio = NAN;
    for (int64 xi = xstart; xi != xend; xi += xinc)
    {
      if (!image.depth.at<float>(yi, xi) || image.depth.at<float>(yi, xi) < 0.05f)
        continue;

      const float depth = image.depth.at<float>(yi, xi);
      const float z = std::abs(depth);
      const float x = ((xi - cx) * z / fx);
      const float ratio = (x - emitter_distance) / z;

      if (!std::isnan(prev_ratio) && ratio * xincf > prev_ratio * xincf)
      {
        // prev_ratio = xe/ze
        // emitter shadow line: x = prev_ratio * z + emitter_distance
        // camera_ratio = xc/zc
        // camera line: x = camera_ratio * z
        // |
        // v
        // intersection at: new_z = emitter_distance / (camera_ratio - prev_ratio)

        const float camera_ratio = x / z;
        const float new_z = emitter_distance / (camera_ratio - prev_ratio);
        result.depth.at<float>(yi, xi) = new_z;
        result.normals.at<cv::Vec3f>(yi, xi) = cv::Vec3f(0.0, 0.0, 0.0);
        result.status.at<uint8>(yi, xi) = ReachingResult::LOST;
      }
      if (depth > 0.0f && image.status.at<uint8>(yi, xi) == ReachingResult::OCCUPIED &&
          (std::isnan(prev_ratio) || ratio * xincf < prev_ratio * xincf))
        prev_ratio = ratio;
    }
  }

  return result;
}

EvaluateViewFilter::ReachingResult EvaluateViewFilter::PixelShift(const ReachingResult & image)
{
  ReachingResult result = image;
  result.depth = 0.0f;
  result.normals = cv::Vec3f(0.0f, 0.0f, 0.0f);
  result.status = ReachingResult::LOST;

  const uint64 width = result.depth.cols;
  const uint64 height = result.depth.rows;

  for (int64 y = 0; y < height; y++)
    for (int64 x = 0; x < width; x++)
    {
      const int64 nx = x - m_pixel_shift_x;
      const int64 ny = y - m_pixel_shift_y;

      if (nx < 0 || ny < 0)
        continue;

      if (nx >= width || ny >= height)
        continue;

      result.depth.at<float>(y, x) = image.depth.at<float>(ny, nx);
      result.status.at<uint8>(y, x) = image.status.at<uint8>(ny, nx);
      result.normals.at<cv::Vec3f>(y, x) = image.normals.at<cv::Vec3f>(ny, nx);
    }

  return result;
}

cv::Mat EvaluateViewFilter::GenerateFilterCircleRectangleMask(const uint64 width, const uint64 height) const
{
  cv::Mat result = cv::Mat(height, width, CV_8UC1, uint8(true));

  for (uint64 y = 0; y < height; y++)
    for (uint64 x = 0; x < width; x++)
    {
      const float radius = std::sqrt(SQR(width / 2.0f) + SQR(height / 2.0f)) * m_circle_radius;
      const float center_x = (m_center_x + 1.0f) * (float(width) / 2.0f);
      const float center_y = (m_center_y + 1.0f) * (float(height) / 2.0f);

      const float dy = float(y) - center_y;
      const float dx = float(x) - center_x;
      const float d = std::sqrt(SQR(dx) + SQR(dy));

      if (!(d < radius))
      {
        result.at<uint8>(y, x) = false;
      }
    }

  for (uint64 y = 0; y < height; y++)
    for (uint64 x = 0; x < width; x++)
    {
      const float ox = m_rectangle_x * width;
      const float oy = m_rectangle_y * height;
      const float w = m_rectangle_width * width;
      const float h = m_rectangle_height * height;

      if (x < ox || y < oy || x >= (ox + w) || y >= (oy + h))
      {
        result.at<uint8>(y, x) = false;
      }
    }

  return result;
}

EvaluateViewFilter::ReachingResult EvaluateViewFilter::FilterCircleRectangle(const ReachingResult &image)
{
  ReachingResult result = image;

  const uint64 width = result.depth.cols;
  const uint64 height = result.depth.rows;

  const cv::Mat mask = GenerateFilterCircleRectangleMask(width, height);

  for (uint64 y = 0; y < height; y++)
    for (uint64 x = 0; x < width; x++)
    {
      if (!mask.at<uint8>(y, x))
      {
        result.depth.at<float>(y, x) = 0.0;
        result.normals.at<cv::Vec3f>(y, x) = cv::Vec3f(0.0, 0.0, 0.0);
        result.status.at<uint8>(y, x) = ReachingResult::LOST;
      }
    }

  return result;
}

EvaluateViewFilter::ReachingResult EvaluateViewFilter::FilterAddRobot(const ReachingResult & image,
                                                                      const cv::Mat & robot_depth_image,
                                                                      const cv::Mat & robot_normal_image)
{
  const uint64 width = robot_depth_image.cols;
  const uint64 height = robot_depth_image.rows;

  ReachingResult result = image;

  for (uint64 y = 0; y < height; y++)
    for (uint64 x = 0; x < width; x++)
    {
      const float rd = robot_depth_image.at<float>(y, x);
      const cv::Vec3f crn = robot_normal_image.at<cv::Vec3f>(y, x);

      if (rd == 0.0f)
        continue;

      const float d = image.depth.at<float>(y, x);

      if (!d || rd < d)
      {
        result.depth.at<float>(y, x) = rd;
        result.normals.at<cv::Vec3f>(y, x) = crn;
        result.status.at<uint8>(y, x) = ReachingResult::OCCUPIED;
      }
    }

  return result;
}

EvaluateViewFilter::ReachingResult EvaluateViewFilter::FilterDiscontinuity(const ReachingResult &image)
{
  const uint64 width = image.depth.cols;
  const uint64 height = image.depth.rows;

  ReachingResult result = image;

  const int64 window = m_discontinuity_filter_window;
  const float threshold = m_discontinuity_filter_th;

  for (int64 y = 0; y < height; y++)
    for (int64 x = 0; x < width; x++)
    {
      const float v = image.depth.at<float>(y, x);
      if (v == 0.0f)
        continue;
      if (image.status.at<uint8>(y, x) != ReachingResult::OCCUPIED)
        continue;

      bool to_be_deleted = false;

      for (int64 dy = -1; dy <= 1; dy++)
        for (int64 dx = -1; dx <= 1; dx++)
        {
          const int64 nx = dx + x;
          const int64 ny = dy + y;

          if (nx < 0 || ny < 0)
            continue;
          if (nx >= width || ny >= height)
            continue;

          if (!dx && !dy)
            continue;
          if (dx * dx + dy * dy > window * window)
            continue;

          const float nv = image.depth.at<float>(ny, nx);
          if (nv == 0.0f)
            continue;
          if (image.status.at<uint8>(ny, nx) != ReachingResult::OCCUPIED)
            continue;

          if (v - nv > threshold)
            to_be_deleted = true;
        }

      if (to_be_deleted)
      {
        for (int64 dy = -window; dy <= window; dy++)
          for (int64 dx = -window; dx <= window; dx++)
          {
            const int64 nx = dx + x;
            const int64 ny = dy + y;

            if (nx < 0 || ny < 0)
              continue;
            if (nx >= width || ny >= height)
              continue;

            if (dx * dx + dy * dy > window * window)
              continue;

            if (image.status.at<uint8>(ny, nx) != ReachingResult::OCCUPIED)
              continue;

            result.status.at<uint8>(ny, nx) = ReachingResult::LOST;
            // leave depth as-is
          }
      }
    }
  return result;
}

EvaluateViewFilter::ReachingResult EvaluateViewFilter::FilterNormalInclination(
    const ReachingResult & image,
    const Eigen::Vector2f & camera_center,
    const Eigen::Vector2f & camera_focal)
{
  ReachingResult result = image;

  const uint64 width = result.depth.cols;
  const uint64 height = result.depth.rows;

  for (uint64 y = 0; y < height; y++)
    for (uint64 x = 0; x < width; x++)
    {
      if (!image.depth.at<float>(y, x))
        continue;
      if (image.status.at<uint8>(y, x) != ReachingResult::OCCUPIED)
        continue;

      const cv::Vec3f n = image.normals.at<cv::Vec3f>(y, x);
      const Eigen::Vector3f en(n[0], n[1], n[2]);

      const Eigen::Vector3f view_ray = Eigen::Vector3f((float(x) - camera_center.x() + 0.5f) / camera_focal.x(),
                                                       (float(y) - camera_center.y() + 0.5f) / camera_focal.y(),
                                                       1.0f
                                                       ).normalized();

      if (en.dot(-view_ray) < std::cos(m_normal_max_angle))
      {
        result.status.at<uint8>(y, x) = ReachingResult::LOST;
        // leave depth as-is
      }
    }

  return result;
}

EvaluateViewFilter::ReachingResult EvaluateViewFilter::FilterMinRange(const ReachingResult &image, const float min_range)
{
  const uint64 width = image.depth.cols;
  const uint64 height = image.depth.rows;

  ReachingResult result = image;

  for (uint64 y = 0; y < height; y++)
    for (uint64 x = 0; x < width; x++)
    {
      if (image.depth.at<float>(y, x) && image.depth.at<float>(y, x) < min_range)
      {
        result.status.at<uint8>(y, x) = ReachingResult::LOST;
      }
    }

  return result;
}

EvaluateViewFilter::ReachingResult EvaluateViewFilter::FilterOpening(const ReachingResult &image, const uint64 opening_size)
{
  const uint64 width = image.depth.cols;
  const uint64 height = image.depth.rows;

  cv::Mat mask = cv::Mat(height, width, CV_8UC1);

  ReachingResult result = image;

  if (opening_size == 0)
    return result;

  for (uint64 y = 0; y < height; y++)
    for (uint64 x = 0; x < width; x++)
      mask.at<uint8>(y, x) = (image.status.at<uint8>(y, x) == ReachingResult::OCCUPIED);

  const uint64 EROSION_SIZE = opening_size;
  cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE,
                                              cv::Size(2 * EROSION_SIZE + 1, 2 * EROSION_SIZE + 1),
                                              cv::Point(EROSION_SIZE, EROSION_SIZE));

  cv::Mat mask_out = mask.clone();
  cv::erode(mask_out, mask_out, element);
  cv::dilate(mask_out, mask_out, element);

  for (uint64 y = 0; y < height; y++)
    for (uint64 x = 0; x < width; x++)
    {
      if (mask.at<uint8>(y, x) && !mask_out.at<uint8>(y, x))
      {
        //result.depth.at<float>(y, x) = 0.0f;
        result.status.at<uint8>(y, x) = ReachingResult::LOST;
      }
    }

  return result;
}

EvaluateViewFilter::ReachingResult EvaluateViewFilter::FilterErosion(const ReachingResult &image, const uint64 erosion_size)
{
  const uint64 width = image.depth.cols;
  const uint64 height = image.depth.rows;

  cv::Mat mask = cv::Mat(height, width, CV_8UC1);

  ReachingResult result = image;

  if (erosion_size == 0)
    return result;

  for (uint64 y = 0; y < height; y++)
    for (uint64 x = 0; x < width; x++)
      mask.at<uint8>(y, x) = (image.status.at<uint8>(y, x) == ReachingResult::OCCUPIED);

  const uint64 EROSION_SIZE = erosion_size;
  cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE,
                                              cv::Size(2 * EROSION_SIZE + 1, 2 * EROSION_SIZE + 1),
                                              cv::Point(EROSION_SIZE, EROSION_SIZE));

  cv::Mat mask_out = mask.clone();
  cv::erode(mask_out, mask_out, element);

  for (uint64 y = 0; y < height; y++)
    for (uint64 x = 0; x < width; x++)
    {
      if (mask.at<uint8>(y, x) && !mask_out.at<uint8>(y, x))
      {
        //result.depth.at<float>(y, x) = 0.0f;
        result.status.at<uint8>(y, x) = ReachingResult::LOST;
      }
    }

  return result;
}

EvaluateViewFilter::ReachingResult EvaluateViewFilter::FilterErosionByDistance(const ReachingResult &image,
                                                                               const uint64 erosion_size,
                                                                               const float depth_scale)
{
  const uint64 width = image.depth.cols;
  const uint64 height = image.depth.rows;

  ReachingResult result = image;

  if (erosion_size == 0)
    return result;
  const int64 EROSION_SIZE = erosion_size;

  for (int64 y = 0; y < height; y++)
    for (int64 x = 0; x < width; x++)
    {
      if (image.status.at<uint8>(y, x) == ReachingResult::OCCUPIED)
        continue;

      const int64 dy = 0;
      for (int64 dx = -0; dx < EROSION_SIZE; dx++)
      {
        const int64 nx = dx + x;
        const int64 ny = dy + y;

        if (dx * dx + dy * dy > EROSION_SIZE * EROSION_SIZE)
          continue;

        if (nx >= width || nx < 0)
          continue;
        if (ny >= height || ny < 0)
          continue;

        if (image.status.at<uint8>(ny, nx) != ReachingResult::OCCUPIED)
          continue;
        float depth = image.depth.at<float>(ny, nx);
        if (depth <= 0.05f)
          continue;

        const int64 new_erosion_size = std::round((depth_scale / depth) * EROSION_SIZE);
        if (dx * dx + dy * dy > new_erosion_size * new_erosion_size)
          continue;

        result.status.at<uint8>(ny, nx) = ReachingResult::LOST;
      }
    }

  return result;
}

EvaluateViewFilter::ReachingResult EvaluateViewFilter::Filter(const EvaluateViewOpenCL::RaycastResult & raycast_result,
                                                              const float voxel_size,
                                                              const Eigen::Vector2f & camera_center,
                                                              const Eigen::Vector2f & camera_focal,
                                                              const float min_range,
                                                              const cv::Mat & robot_depth_image,
                                                              const cv::Mat & robot_normal_image)
{
  const uint64 width = raycast_result.size.x();
  const uint64 height = raycast_result.size.y();

  ReachingResult result(width, height);

  for (uint64 y = 0; y < height; y++)
    for (uint64 x = 0; x < width; x++)
    {
      const EvaluateViewOpenCL::CellResult & ray = raycast_result.ray_results[raycast_result.size.x() * y + x];
      
      result.status.at<uint8>(y, x) = ray.status;
      result.depth.at<float>(y, x) = ray.z * voxel_size * ray.local_direction.z();
      result.normals.at<cv::Vec3f>(y, x) = cv::Vec3f(ray.normal.x(), ray.normal.y(), ray.normal.z());
    }

  result = PixelShift(result);

  result = FilterAddRobot(result, robot_depth_image, robot_normal_image);

  result = FilterCircleRectangle(result);
  //result = FilterDiscontinuity(result);

  result = FilterNormalInclination(result, camera_center, camera_focal); // maybe probabilistic?
  result = FilterErosionByDistance(result, m_erosion_kernel_size, m_erosion_depth_scale);
  result = ShadowRemoval(camera_center, camera_focal, m_shadow_removal_emitter_distance_left, result);
  result = ShadowRemoval(camera_center, camera_focal, m_shadow_removal_emitter_distance_right, result);
  result = FilterMinRange(result, min_range);
  result = FilterOpening(result, m_opening_kernel_size);

  return result;
}
