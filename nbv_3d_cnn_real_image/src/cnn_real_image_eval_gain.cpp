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

#include "cnn_real_image_eval_gain.h"

#include <ros/ros.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/JointState.h>
#include <actionlib/client/simple_action_client.h>
#include <message_serialization/serialize.h>
#include <message_serialization/sensor_msgs_yaml.h>
#include <eigen_conversions/eigen_msg.h>
#include <cv_bridge/cv_bridge.h>
#include <pcl_conversions/pcl_conversions.h>

#include <rmonica_voxelgrid_common/voxelgrid.h>
#include <rmonica_voxelgrid_common/metadata.h>
#include <nbv_3d_cnn_evaluate_view_msgs/EvaluateViewAction.h>
#include <nbv_3d_cnn_evaluate_view_msgs/SetEnvironmentAction.h>

#include <string>
#include <memory>
#include <sstream>
#include <stdint.h>
#include <fstream>

#include <Eigen/Dense>
#include <Eigen/StdVector>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>

class CNNRealImageEvalGain
{
  public:
  typedef uint64_t uint64;
  typedef uint8_t uint8;

  enum class Mode
  {
    NONE,
    FIXED,
    ADHOC,
    ONLY2D,
    ONLY3D,
    NOATTENTION,
    UNFROZEN,
    FULL
  };

  template <typename T>
  static T SQR(const T & t) {return t * t; }

  Eigen::Vector3f StringToVector3f(const std::string & str)
  {
    Eigen::Vector3f result;
    std::istringstream istr(str);
    istr >> result.x() >> result.y() >> result.z();
    if (!istr)
    {
      ROS_FATAL("cnn_real_image_evaluate: unable to parse vector3f: %s", str.c_str());
      exit(1);
    }
    return result;
  }

  CNNRealImageEvalGain(ros::NodeHandle & nh): m_nh(nh)
  {
    std::string param_string;
    int param_int;
    double param_double;

    m_timer = m_nh.createTimer(ros::Duration(0.0001), &CNNRealImageEvalGain::onTimer, this, false);

    m_nh.param<std::string>(PARAM_NAME_MODE, param_string, PARAM_DEFAULT_MODE);
    if (param_string == PARAM_VALUE_MODE_NONE)
      m_mode = Mode::NONE;
    else if (param_string == PARAM_VALUE_MODE_ADHOC)
      m_mode = Mode::ADHOC;
    else if (param_string == PARAM_VALUE_MODE_ONLY2D)
      m_mode = Mode::ONLY2D;
    else if (param_string == PARAM_VALUE_MODE_FIXED)
      m_mode = Mode::FIXED;
    else if (param_string == PARAM_VALUE_MODE_FULL)
      m_mode = Mode::FULL;
    else if (param_string == PARAM_VALUE_MODE_ONLY3D)
      m_mode = Mode::ONLY3D;
    else if (param_string == PARAM_VALUE_MODE_NOATTENTION)
      m_mode = Mode::NOATTENTION;
    else if (param_string == PARAM_VALUE_MODE_UNFROZEN)
      m_mode = Mode::UNFROZEN;
    else
    {
      ROS_FATAL("cnn_real_image_eval_gain: unknown mode: %s", param_string.c_str());
      exit(1);
    }

    m_nh.param<std::string>(PARAM_NAME_GT_EVAL_FILE_PREFIX, m_gt_eval_file_prefix, PARAM_DEFAULT_GT_EVAL_FILE_PREFIX);
    m_nh.param<std::string>(PARAM_NAME_SCENARIO_FILE_PREFIX, m_scenario_file_prefix, PARAM_DEFAULT_SCENARIO_FILE_PREFIX);
    m_nh.param<std::string>(PARAM_NAME_EVALUATION_FILE_PREFIX, m_evaluation_file_prefix, PARAM_DEFAULT_EVALUATION_FILE_PREFIX);
    m_nh.param<std::string>(PARAM_NAME_VARIANT_FILE_PREFIX, m_environment_file_prefix, PARAM_DEFAULT_VARIANT_FILE_PREFIX);
    m_nh.param<std::string>(PARAM_NAME_EVAL_GAIN_FILE_PREFIX, m_eval_gain_file_prefix, PARAM_DEFAULT_EVAL_GAIN_FILE_PREFIX);

    m_nh.param<std::string>(PARAM_NAME_MASK_FILE_NAME, m_mask_file_name, PARAM_DEFAULT_MASK_FILE_NAME);

    m_nh.param<int>(PARAM_NAME_SCENARIO_FIRST_INDEX, param_int, PARAM_DEFAULT_SCENARIO_FIRST_INDEX);
    m_scenario_first_index = param_int;
    m_nh.param<int>(PARAM_NAME_SCENARIO_LAST_INDEX, param_int, PARAM_DEFAULT_SCENARIO_LAST_INDEX);
    m_scenario_last_index = param_int;

    m_nh.param<bool>(PARAM_NAME_SAVE_IMAGES, m_save_images, PARAM_DEFAULT_SAVE_IMAGES);

    m_current_scenario = m_scenario_first_index;
    m_current_environment_variant = 0;
    m_current_image = 0;

    m_first_iter_scenario = true;
    m_mask_loaded = false;
    m_first_iter_environment_variant = false;
  }

  struct CompareMaskImagesResult
  {
    float fp = 0.0f;
    float fn = 0.0f;
    float tp = 0.0f;
    float tn = 0.0f;
    float rmse = 0.0f;
    cv::Mat diff_image;
  };

  CompareMaskImagesResult CompareMaskImages(const cv::Mat & image,
                                            const cv::Mat & ground_truth,
                                            const cv::Mat & mask
                                            )
  {
    const uint64 width = image.cols;
    const uint64 height = image.rows;

    CompareMaskImagesResult result;

    result.diff_image = cv::Mat(height, width, CV_8UC3, cv::Vec3b(0, 0, 0));

    uint64 counter = 0;
    result.rmse = 0.0f;

    for (uint64 y = 0; y < height; y++)
      for (uint64 x = 0; x < width; x++)
      {
        if (!mask.at<float>(y, x))
          continue;

        const float v = image.at<float>(y, x);
        const float gt = ground_truth.at<float>(y, x);
        cv::Vec3b & color = result.diff_image.at<cv::Vec3b>(y, x);

        result.rmse += SQR(v - gt);
        counter++;

        if (!gt)
        {
          result.fp += v;
          result.tn += 1.0f - v;

          color[2] = 255 * v;
        }
        else
        {
          result.tp += v;
          result.fn += 1.0f - v;

          color[0] = 255 * (1.0 - v);
        }
      }
    if (counter)
    {
      result.rmse /= counter;
      result.rmse = std::sqrt(result.rmse);
    }

    return result;
  }

  uint64 GetMaxExistingFile(const std::string & prefix, const std::string & suffix)
  {
    uint64 result = 0;
    const uint64 SANITY = 1000000;

    for (result = 0; result < SANITY; result++)
    {
      const std::string filename = prefix + std::to_string(result) + suffix;
      std::ifstream ifile(filename.c_str());
      if (!ifile)
        break;
    }

    return result;
  }

  void CompareScoreImages(const cv::Mat & image,
                          const cv::Mat & ground_truth,
                          const cv::Mat & mask,
                          float & rmse_error,
                          float & abs_error,
                          float & total_error
                          )
  {
    const uint64 width = image.cols;
    const uint64 height = image.rows;

    uint64 counter = 0;
    float mse_error = 0.0f;
    abs_error = 0.0f;
    double gt_total = 0.0f;
    double image_total = 0.0f;
    for (uint64 y = 0; y < height; y++)
      for (uint64 x = 0; x < width; x++)
      {
        if (!mask.at<float>(y, x))
          continue;
        counter++;
        const float v = image.at<float>(y, x);
        const float gt = ground_truth.at<float>(y, x);
        gt_total += gt;
        image_total += v;
        const float diff = v - gt;
        mse_error += SQR(diff);
        abs_error += std::abs(diff);
      }
    rmse_error = std::sqrt(mse_error);
    abs_error /= float(counter);
    total_error = std::abs(gt_total - image_total);
  }

  void UpdateEvaluationFile(const uint64 scenario,
                            const uint64 variant,
                            const uint64 image,
                            CompareMaskImagesResult r,
                            const float scoreRMSE,
                            const float scoreABSE,
                            const float totalE
                            )
  {
    if (!m_evaluation_file)
    {
      const std::string log_file_name = m_eval_gain_file_prefix + "aaa_scores.csv";
      ROS_INFO("cnn_real_image_eval_gain: saving to log file %s", log_file_name.c_str());
      m_evaluation_file.reset(new std::ofstream(log_file_name.c_str()));
      if (!*m_evaluation_file)
      {
        ROS_FATAL("cnn_real_image_eval_gain: could not create evaluation file %s", log_file_name.c_str());
        exit(1);
      }

      (*m_evaluation_file) << "Scn" << "\t" << "Image" << "\t" << "Variant" << "\t"
                           << "FP" << "\t" << "FN" << "\t" << "TP" << "\t" << "TN" << "\t"
                           << "RMSE" << "\t"
                           << "ScoreRMSE" << "\t" << "ScoreABSE" << "\t" << "TotalE" << "\n";
    }

    ROS_INFO("cnn_real_image_eval_gain: updating evaluation file.");
    (*m_evaluation_file) << scenario << "\t" << image << "\t" << variant << "\t"
                         << r.fp << "\t" << r.fn << "\t" << r.tp << "\t" << r.tn << "\t"
                         << r.rmse << "\t"
                         << scoreRMSE << "\t" << scoreABSE << "\t" << totalE << "\n";
    if (image % 50 == 0)
      (*m_evaluation_file) << std::flush;
  }

  void onTimer(const ros::TimerEvent &)
  {
    ROS_INFO("cnn_real_image_evaluate: processin start, scenario %u, image %u, variant %u.",
             unsigned(m_current_scenario), unsigned(m_current_image), unsigned(m_current_environment_variant));

    if (!m_mask_loaded)
    {
      ROS_INFO("cnn_real_image_evaluate: loading mask %s.", m_mask_file_name.c_str());
      m_mask = cv::imread(m_mask_file_name, cv::IMREAD_GRAYSCALE);
      if (!m_mask.data)
      {
        ROS_FATAL("cnn_real_image_evaluate: could not load mask %s.", m_mask_file_name.c_str());
        exit(1);
      }
      m_mask.convertTo(m_mask, CV_32FC1);
      m_mask_loaded = true;
    }

    const std::string scenario_str = std::to_string(m_current_scenario);

    if (m_first_iter_scenario)
    {
      ROS_INFO("cnn_real_image_evaluate: switching to scenario %u", unsigned(m_current_scenario));
      const std::string image_prefix = m_gt_eval_file_prefix + "mask_" + scenario_str + "_";
      const std::string image_suffix = "_0.png";
      m_max_images = GetMaxExistingFile(image_prefix, image_suffix);
      ROS_INFO("cnn_real_image_evaluate: scenario is %u, found %u images.",
               unsigned(m_current_scenario), unsigned(m_max_images));

      const std::string variant_prefix = m_scenario_file_prefix + scenario_str + m_environment_file_prefix + "partial_";
      const std::string variant_suffix = ".voxelgrid";
      m_max_environment_variants = GetMaxExistingFile(variant_prefix, variant_suffix);
      ROS_INFO("cnn_real_image_evaluate: scenario is %u, found %u environment variants.",
               unsigned(m_current_scenario), unsigned(m_max_environment_variants));

      m_first_iter_scenario = false;
      m_current_image = 0;
      m_current_environment_variant = 0;
      m_first_iter_environment_variant = true;
    }

    const std::string environment_variant_str = std::to_string(m_current_environment_variant);

    if (m_first_iter_environment_variant)
    {
      ROS_INFO("cnn_real_image_evaluate: switching to environment variant %u", unsigned(m_current_environment_variant));

      m_first_iter_environment_variant = false;
      m_current_image = 0;
    }

    const std::string image_str = std::to_string(m_current_image);

    const std::string gt_scores_filename = m_gt_eval_file_prefix + "score_" + scenario_str + "_" +
                                           image_str + "_" + environment_variant_str + ".png";
    ROS_INFO("cnn_real_image_evaluate: loading ground truth scores image %s", gt_scores_filename.c_str());
    cv::Mat gt_scores_image = cv::imread(gt_scores_filename, cv::IMREAD_ANYDEPTH);
    gt_scores_image.convertTo(gt_scores_image, CV_32FC1, 1.0 / (10.0 * 1000.0));
    if (!gt_scores_image.data)
    {
      ROS_FATAL("cnn_real_image_eval_gain: could not load image %s", gt_scores_filename.c_str());
      exit(1);
    }

    const std::string gt_mask_filename = m_gt_eval_file_prefix + "mask_" + scenario_str + "_" +
                                         image_str + "_" + environment_variant_str + ".png";
    ROS_INFO("cnn_real_image_evaluate: loading ground truth mask image %s", gt_mask_filename.c_str());
    cv::Mat gt_mask_image = cv::imread(gt_mask_filename, cv::IMREAD_ANYDEPTH);
    gt_mask_image.convertTo(gt_mask_image, CV_32FC1, 1.0 / 255.0);
    if (!gt_mask_image.data)
    {
      ROS_FATAL("cnn_real_image_eval_gain: could not load image %s", gt_mask_filename.c_str());
      exit(1);
    }

    const std::string mask_filename = m_evaluation_file_prefix + "mask_" +
        scenario_str + "_" + image_str + "_" + environment_variant_str + ".png";
    ROS_INFO("cnn_real_image_evaluate: loading mask image %s", mask_filename.c_str());
    cv::Mat mask_image = cv::imread(mask_filename, cv::IMREAD_ANYDEPTH);
    mask_image.convertTo(mask_image, CV_32FC1, 1.0 / 255.0);
    if (!mask_image.data)
    {
      ROS_FATAL("cnn_real_image_eval_gain: could not load image %s", gt_mask_filename.c_str());
      exit(1);
    }

    const std::string score_filename = m_evaluation_file_prefix + "score_" +
        scenario_str + "_" + image_str + "_" + environment_variant_str + ".png";
    ROS_INFO("cnn_real_image_evaluate: loading score image %s", score_filename.c_str());
    cv::Mat scores_image = cv::imread(score_filename, cv::IMREAD_ANYDEPTH);
    scores_image.convertTo(scores_image, CV_32FC1, 1.0 / (10.0 * 1000.0));
    if (!scores_image.data)
    {
      ROS_FATAL("cnn_real_image_eval_gain: could not load image %s", gt_scores_filename.c_str());
      exit(1);
    }

    float rmse_error, abse_error, total_error;
    CompareScoreImages(scores_image, gt_scores_image, m_mask,
                       rmse_error, abse_error, total_error);
    ROS_INFO("cnn_real_image_eval_gain: RMSE error: %f", float(rmse_error));
    ROS_INFO("cnn_real_image_eval_gain: ABSE error: %f", float(abse_error));
    ROS_INFO("cnn_real_image_eval_gain: Total error: %f", float(total_error));

    const CompareMaskImagesResult r = CompareMaskImages(mask_image, gt_mask_image, m_mask);

    UpdateEvaluationFile(m_current_scenario, m_current_environment_variant, m_current_image,
                         r, rmse_error, abse_error, total_error);


    if (m_save_images)
    {
      {
        const std::string cmp_filename = m_eval_gain_file_prefix + "score_cmp_" +
                                         scenario_str + "_" + image_str + "_" + environment_variant_str + ".png";
        ROS_INFO("cnn_real_image_eval_gain: saving cmp to file %s.", cmp_filename.c_str());
        cv::Mat cmp_image;
        cv::hconcat(scores_image, gt_scores_image, cmp_image);
        cv::Mat cmp_image_int;
        cmp_image.convertTo(cmp_image_int, CV_16UC1, 10.0 * 1000.0);
        cv::imwrite(cmp_filename, cmp_image_int);
      }

      {
        const std::string error_filename = m_eval_gain_file_prefix + "error_" +
                                           scenario_str + "_" + image_str + "_" + environment_variant_str + ".png";
        cv::imwrite(error_filename, r.diff_image);
      }

      {
        const std::string score_error_filename = m_eval_gain_file_prefix + "score_error_" +
                                             scenario_str + "_" + image_str + "_" + environment_variant_str + ".png";
        ROS_INFO("cnn_real_image_eval_gain: saving mask to file %s.", score_error_filename.c_str());
        cv::Mat score_error = cv::Mat(scores_image.rows, scores_image.cols, CV_32FC3);
        score_error = cv::Vec3f(0.0f, 0.0f, 0.0f);

        for (uint64 y = 0; y < scores_image.rows; y++)
          for (uint64 x = 0; x < scores_image.cols; x++)
          {
            const float gt = gt_scores_image.at<float>(y, x);
            const float v = scores_image.at<float>(y, x);
            if (gt - v > 0.0)
            {
              score_error.at<cv::Vec3f>(y, x)[0] = gt - v; // false negative is blue
            }
            if (v - gt > 0.0)
            {
              score_error.at<cv::Vec3f>(y, x)[2] = v - gt; // false positive is red
            }
          }


        cv::Mat score_error_int;
        score_error.convertTo(score_error_int, CV_8UC3, 64.0f);
        cv::imwrite(score_error_filename, score_error_int);
      }
    }

    m_current_image++;
    if (m_current_image >= m_max_images)
    {
      m_current_environment_variant++;
      ROS_INFO("cnn_real_image_evaluate: current env. variant is now %u", unsigned(m_current_environment_variant));
      m_current_image = 0;
      m_first_iter_environment_variant = true;
    }

    if (m_current_environment_variant >= m_max_environment_variants)
    {
      m_current_scenario++;
      m_first_iter_scenario = true;
    }

    if (m_current_scenario >= m_scenario_last_index)
    {
      ROS_INFO("cnn_real_image_evaluate: all scenarios processed, end.");
      m_timer.stop();

      if (m_evaluation_file)
        m_evaluation_file.reset();
    }
  }

  private:
  ros::NodeHandle & m_nh;

  ros::Timer m_timer;

  uint64 m_current_scenario;
  bool m_first_iter_scenario;
  uint64 m_current_environment_variant;
  bool m_first_iter_environment_variant;
  uint64 m_current_image;

  uint64 m_max_images;
  uint64 m_max_environment_variants;

  std::string m_mask_file_name;
  bool m_mask_loaded;
  cv::Mat m_mask;

  bool m_save_images;

  uint64 m_scenario_first_index;
  uint64 m_scenario_last_index;
  std::string m_scenario_file_prefix;

  std::string m_gt_eval_file_prefix;
  std::string m_environment_file_prefix;
  std::string m_image_file_prefix;

  std::string m_evaluation_file_prefix;
  std::shared_ptr<std::ofstream> m_evaluation_file;

  std::string m_eval_gain_file_prefix;

  Mode m_mode;
};


int main(int argc, char ** argv)
{
  ros::init(argc, argv, "cnn_real_image_eval_gain");
  ros::NodeHandle nh("~");

  CNNRealImageEvalGain crieg(nh);

  ros::spin();

  return 0;
}

