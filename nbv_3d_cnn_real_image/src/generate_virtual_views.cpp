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

#include "generate_virtual_views.h"

// ROS
#include <ros/ros.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <actionlib/client/simple_action_client.h>
#include <nbv_3d_cnn_evaluate_view_msgs/SetEnvironmentAction.h>
#include <nbv_3d_cnn_evaluate_view_msgs/RawProjectionAction.h>
#include <nbv_3d_cnn_evaluate_view_msgs/GenerateFilterCircleRectangleMaskAction.h>
#include <nbv_3d_cnn_evaluate_view_msgs/GroundTruthEvaluateViewAction.h>
#include <eigen_conversions/eigen_msg.h>
#include <cv_bridge/cv_bridge.h>
#include <pcl_conversions/pcl_conversions.h>

// custom
#include <rmonica_voxelgrid_common/voxelgrid.h>
#include <rmonica_voxelgrid_common/metadata.h>

// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

// PCL
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>

#include <string>
#include <fstream>
#include <sstream>
#include <stdint.h>

// Serialization headers
#include <message_serialization/serialize.h>
// Datatype-specific serialization header
#include <message_serialization/sensor_msgs_yaml.h>

class GenerateVirtualViews
{
  public:
  typedef rmonica_voxelgrid_common::Voxelgrid Voxelgrid;

  typedef actionlib::SimpleActionClient<nbv_3d_cnn_evaluate_view_msgs::SetEnvironmentAction> SetEnvironmentActionClient;
  typedef std::shared_ptr<SetEnvironmentActionClient> SetEnvironmentActionClientPtr;
  typedef actionlib::SimpleActionClient<nbv_3d_cnn_evaluate_view_msgs::RawProjectionAction> RawProjectionActionClient;
  typedef std::shared_ptr<RawProjectionActionClient> RawProjectionActionClientPtr;
  typedef actionlib::SimpleActionClient<nbv_3d_cnn_evaluate_view_msgs::GenerateFilterCircleRectangleMaskAction>
    GenerateFilterCircleRectangleMaskActionClient;
  typedef std::shared_ptr<GenerateFilterCircleRectangleMaskActionClient> GenerateFilterCircleRectangleMaskActionClientPtr;
  typedef actionlib::SimpleActionClient<nbv_3d_cnn_evaluate_view_msgs::GroundTruthEvaluateViewAction>
    GroundTruthEvaluateViewActionClient;
  typedef std::shared_ptr<GroundTruthEvaluateViewActionClient> GroundTruthEvaluateViewActionClientPtr;

  typedef std::vector<float> FloatVector;
  typedef std::vector<double> DoubleVector;

  typedef pcl::PointCloud<pcl::PointXYZI> PointXYZICloud;
  typedef pcl::PointCloud<pcl::PointXYZ> PointXYZCloud;

  typedef uint32_t uint32;
  typedef uint64_t uint64;
  typedef uint8_t uint8;
  typedef uint16_t uint16;

  enum
  {
    CLEAR_POI_REGION_NONE = 0,
    CLEAR_POI_REGION_YP   = 1,
    CLEAR_POI_REGION_YN   = 2,
    CLEAR_POI_REGION_ALL  = 3,
  };

  typedef VoxelgridMetadata::Metadata Metadata;
  typedef VoxelgridMetadata::MetadataPtr MetadataPtr;

  GenerateVirtualViews(ros::NodeHandle & nh): m_nh(nh)
  {
    std::string param_string;
    int param_int;
    double param_double;
    m_nh.param<std::string>(PARAM_NAME_VOXELGRID_PREFIX, param_string, PARAM_DEFAULT_VOXELGRID_PREFIX);
    m_voxelgrid_prefix = param_string;

    m_nh.param<std::string>(PARAM_NAME_TSDF_VOLUME_FILENAME, param_string, PARAM_DEFAULT_TSDF_VOLUME_FILENAME);
    m_tsdf_volume_filename = param_string;

    m_nh.param<std::string>(PARAM_NAME_VOXELGRID_METADATA_FILENAME, param_string, PARAM_DEFAULT_VOXELGRID_METADATA_FILENAME);
    m_voxelgrid_metadata_filename = param_string;

    m_nh.param<std::string>(PARAM_NAME_INPUT_IMAGE_PREFIX, param_string, PARAM_DEFAULT_INPUT_IMAGE_PREFIX);
    m_input_image_prefix = param_string;

    m_nh.param<std::string>(PARAM_NAME_OUTPUT_IMAGE_PREFIX, param_string, PARAM_DEFAULT_OUTPUT_IMAGE_PREFIX);
    m_output_image_prefix = param_string;

    m_nh.param<int>(PARAM_NAME_IMAGE_NUMBER, param_int, PARAM_DEFAULT_IMAGE_NUMBER);
    m_image_number = param_int;

    m_nh.param<int>(PARAM_NAME_IMAGE_NUMBER_COUNT, param_int, PARAM_DEFAULT_IMAGE_NUMBER_COUNT);
    m_image_number_count = param_int;

    m_nh.param<int>(PARAM_NAME_DEPTH_IMAGES_PER_POSE, param_int, PARAM_DEFAULT_DEPTH_IMAGES_PER_POSE);
    m_depth_images_per_pose = param_int;

    m_nh.param<double>(PARAM_NAME_MAX_RANGE, param_double, PARAM_DEFAULT_MAX_RANGE);
    m_max_range = param_double;

    m_nh.param<double>(PARAM_NAME_MIN_RANGE, param_double, PARAM_DEFAULT_MIN_RANGE);
    m_min_range = param_double;

    m_nh.param<std::string>(PARAM_NAME_POI_FILE_NAME, m_poi_file_name, PARAM_DEFAULT_POI_FILE_NAME);

    m_image_number_i = 0;
    m_voxelgrid_i = 0;

    m_set_environment_ac.reset(new SetEnvironmentActionClient("/evaluate_view/set_environment", true));
    m_raw_projection_ac.reset(new RawProjectionActionClient("/evaluate_view/raw_projection", true));
    m_filter_circle_rectangle_mask_ac.reset(new GenerateFilterCircleRectangleMaskActionClient(
                                              "/evaluate_view/generate_filter_circle_rectangle_mask", true));
    m_ground_truth_evaluate_view_ac.reset(new GroundTruthEvaluateViewActionClient(
                                            "/evaluate_view/ground_truth_evaluate_view", true));

    ROS_INFO("generate_virtual_views: waiting for server...");

    m_set_environment_ac->waitForServer();
    m_raw_projection_ac->waitForServer();
    m_filter_circle_rectangle_mask_ac->waitForServer();
    m_ground_truth_evaluate_view_ac->waitForServer();

    ROS_INFO("generate_virtual_views: done.");

    m_timer = m_nh.createTimer(ros::Duration(1.0), &GenerateVirtualViews::onTimer, this, true);
  }

  std::string type2str(int type) {
    std::string r;

    uint8_t depth = type & CV_MAT_DEPTH_MASK;
    uint8_t chans = 1 + (type >> CV_CN_SHIFT);

    switch ( depth ) {
      case CV_8U:  r = "8U"; break;
      case CV_8S:  r = "8S"; break;
      case CV_16U: r = "16U"; break;
      case CV_16S: r = "16S"; break;
      case CV_32S: r = "32S"; break;
      case CV_32F: r = "32F"; break;
      case CV_64F: r = "64F"; break;
      default:     r = "User"; break;
    }

    r += "C";
    r += (chans+'0');

    return r;
  }

  void LoadVoxelgrid(const std::string & voxelgrid_filename, const std::string & metadata_filename,
                     const std::string & tsdf_volume_filename)
  {
    ROS_INFO("generate_virtual_views: loading voxelgrid from file %s", voxelgrid_filename.c_str());
    const Voxelgrid::Ptr vg = Voxelgrid::FromFileTernaryBinary(voxelgrid_filename);
    if (!vg)
    {
      ROS_ERROR("generate_virtual_views: could not load voxelgrid.");
      exit(1);
    }

    const Eigen::Vector3i voxelgrid_size = vg->GetSize();

    ROS_INFO("generate_virtual_views: loading voxelgrid metadata from file %s", metadata_filename.c_str());
    MetadataPtr metadata = VoxelgridMetadata::LoadMetadata(metadata_filename);
    if (!metadata)
    {
      ROS_ERROR("generate_virtual_views: could not load grid metadata!");
      exit(1);
    }

    ROS_INFO("generate_virtual_views: loading TSDF from file %s", tsdf_volume_filename.c_str());
    PointXYZICloud tsdf_cloud;
    if (pcl::io::loadPCDFile(tsdf_volume_filename, tsdf_cloud) < 0)
    {
      ROS_ERROR("generate_virtual_views: could not load TSDF cloud!");
      exit(1);
    }
    ROS_INFO("generate_virtual_views: TSDF cloud has size %u", unsigned(tsdf_cloud.size()));

    const float voxel_size = metadata->voxel_size;

    Voxelgrid vg2 = *vg;

    {
      nbv_3d_cnn_evaluate_view_msgs::SetEnvironmentGoal se_goal;
      se_goal.voxel_size = voxel_size;

      se_goal.environment_origin.x = metadata->bbox_min.x();
      se_goal.environment_origin.y = metadata->bbox_min.y();
      se_goal.environment_origin.z = metadata->bbox_min.z();

      se_goal.environment = vg2.ToInt8MultiArray();

      se_goal.probabilities_voxel_size = m_cnn_matrix_voxel_size;

      se_goal.probabilities_origin[0] = 0;
      se_goal.probabilities_origin[1] = 0;
      se_goal.probabilities_origin[2] = 0;
      se_goal.probabilities_size[0] = 0;
      se_goal.probabilities_size[1] = 0;
      se_goal.probabilities_size[2] = 0;

      pcl::toROSMsg(tsdf_cloud, se_goal.tsdf_cloud);

      ROS_INFO("generate_virtual_views: setting environment...");
      m_set_environment_ac->sendGoal(se_goal);
      m_set_environment_ac->waitForResult();
    }

    ROS_INFO_STREAM("generate_virtual_views: bounding box is " << metadata->bbox_max.transpose() << " - "
                    << metadata->bbox_max.transpose());
    ROS_INFO_STREAM("generate_virtual_views: voxel size is " << voxel_size);
  }

  void SaveCircleRectangleMask(const uint64 width, const uint64 height)
  {
    ROS_INFO("generate_virtual_views: SaveCircleRectangleMask...");
    nbv_3d_cnn_evaluate_view_msgs::GenerateFilterCircleRectangleMaskGoal goal;
    goal.height = height;
    goal.width = width;

    m_filter_circle_rectangle_mask_ac->sendGoal(goal);
    m_filter_circle_rectangle_mask_ac->waitForResult();
    nbv_3d_cnn_evaluate_view_msgs::GenerateFilterCircleRectangleMaskResultConstPtr result =
        m_filter_circle_rectangle_mask_ac->getResult();

    cv_bridge::CvImagePtr cv_bridge(new cv_bridge::CvImage);

    cv_bridge = cv_bridge::toCvCopy(result->mask);

    const std::string filename = m_output_image_prefix + "mask.png";
    ROS_INFO("generate_virtual_views: saving mask to %s", filename.c_str());
    if (!cv::imwrite(filename, cv_bridge->image))
    {
      ROS_FATAL("generate_virtual_views: could not save mask to %s", filename.c_str());
      exit(1);
    }


    ROS_INFO("generate_virtual_views: SaveCircleRectangleMask end.");
  }

  bool LoadPose(const std::string & filename, Eigen::Affine3f & pose)
  {
    std::ifstream ifile(filename.c_str());
    for (uint64 y = 0; y < 3; y++)
    {
      for (uint64 x = 0; x < 3; x++)
        ifile >> pose.linear()(y, x);
      ifile >> pose.translation()[y];
    }
    if (!ifile)
      return false;
    return true;
  }

  bool RawProjection(const uint64 number,
                     cv::Mat & predicted_depth_image,
                     cv::Mat & predicted_normal_image,
                     cv::Mat & predicted_robot_image,
                     cv::Mat & predicted_robot_normal_image)
  {
    cv_bridge::CvImagePtr cv_bridge(new cv_bridge::CvImage);

    ROS_INFO("generate_virtual_views: preparing evaluation...");

    const std::string image_number_str = std::to_string(number);

    const std::string camera_info_filename = m_input_image_prefix + "camera_info_" + image_number_str + ".txt";
    ROS_INFO("generate_virtual_views: loading camera info: %s", camera_info_filename.c_str());
    sensor_msgs::CameraInfo camera_info;
    if (!message_serialization::deserialize(camera_info_filename, camera_info))
    {
      ROS_ERROR("generate_virtual_views: could not open or deserialize file!");
      return false;
    }

    Eigen::Affine3f pose;
    {
      const std::string pose_filename = m_input_image_prefix + "pose_" + image_number_str + ".matrix";
      ROS_INFO("generate_virtual_views: loading pose from %s", pose_filename.c_str());
      if (!LoadPose(pose_filename, pose))
        return false;
      ROS_INFO_STREAM("generate_virtual_views: loaded pose " << pose.matrix() << "\n");
    }

    sensor_msgs::JointState joint_state;
    {
      const std::string joint_state_filename = m_input_image_prefix + "joint_state_" + image_number_str + ".txt";
      ROS_INFO("generate_virtual_views: loading joint state from %s", joint_state_filename.c_str());
      std::ifstream ifile(joint_state_filename.c_str());
      DoubleVector js;
      for (uint64 i = 0; i < 6; i++)
      {
        double v;
        ifile >> v;
        js.push_back(v);
      }
      if (!ifile)
      {
        ROS_ERROR("generate_virtual_views: load joint state failed!");
        return false;
      }

      joint_state.position = js;
      joint_state.name = {"joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"};
    }

    nbv_3d_cnn_evaluate_view_msgs::RawProjectionGoal ev_goal;
    ev_goal.center_x = camera_info.K[2];
    ev_goal.center_y = camera_info.K[5];
    ev_goal.focal_x = camera_info.K[0];
    ev_goal.focal_y = camera_info.K[4];
    ev_goal.size_x = camera_info.width;
    ev_goal.size_y = camera_info.height;

    ev_goal.joint_state = joint_state;

    ev_goal.max_range = 4.0;
    ev_goal.min_range = 0.5;

    ev_goal.predict_robot_image = true;

    tf::poseEigenToMsg(pose.cast<double>(), ev_goal.pose);

    ROS_INFO("generate_virtual_views: sending evaluation goal...");
    m_raw_projection_ac->sendGoal(ev_goal);
    m_raw_projection_ac->waitForResult();

    nbv_3d_cnn_evaluate_view_msgs::RawProjectionResultConstPtr ev_result = m_raw_projection_ac->getResult();

    {
      cv_bridge = cv_bridge::toCvCopy(ev_result->predicted_depth_image);
      predicted_depth_image = cv_bridge->image;

      cv_bridge = cv_bridge::toCvCopy(ev_result->predicted_normal_image);
      predicted_normal_image = cv_bridge->image;

      cv_bridge = cv_bridge::toCvCopy(ev_result->predicted_robot_image);
      predicted_robot_image = cv_bridge->image;

      cv_bridge = cv_bridge::toCvCopy(ev_result->predicted_robot_normal_image);
      predicted_robot_normal_image = cv_bridge->image;
    }

    return true;
  }

  bool GetGroundTruth(const uint64 number, const uint64 environment_number)
  {
    const std::string image_number_str = std::to_string(number);
    const std::string environment_number_str = std::to_string(environment_number);

    cv::Mat is_pixel_present_prob;
    cv::Mat max_depth;
    for (uint64 depth_image_i = 0; depth_image_i < m_depth_images_per_pose; depth_image_i++)
    {
      const std::string sample_number_str = std::to_string(depth_image_i);

      const std::string filename = m_input_image_prefix + "depth_image_" + image_number_str + "_" + sample_number_str + ".png";
      ROS_INFO("generate_virtual_views: loading file %s", filename.c_str());

      cv::Mat depth_image = cv::imread(filename, cv::IMREAD_ANYDEPTH);
      if (!depth_image.data)
      {
        ROS_ERROR("generate_virtual_views: could not load image: %s", filename.c_str());
        return false;
      }

      if (depth_image_i == 0)
      {
        is_pixel_present_prob = cv::Mat(depth_image.rows, depth_image.cols, CV_32FC1, 0.0f);
        max_depth = depth_image;
      }

      for (uint64 y = 0; y < depth_image.rows; y++)
        for (uint64 x = 0; x < depth_image.cols; x++)
        {
          if (depth_image.at<uint16>(y, x) != 0)
            is_pixel_present_prob.at<float>(y, x) = 1.0f;
        }

      cv::max(max_depth, depth_image, max_depth);
    }
    //is_pixel_present_prob /= float(m_depth_images_per_pose);

    is_pixel_present_prob.convertTo(is_pixel_present_prob, CV_8UC1, 255);

    if (environment_number == 0)
    {
      const std::string prob_filename = m_output_image_prefix + "gt_" + image_number_str + ".png";
      ROS_INFO("generate_virtual_views: saving gt file %s", prob_filename.c_str());
      if (!cv::imwrite(prob_filename, is_pixel_present_prob))
      {
        ROS_ERROR("generate_virtual_views: could not save file %s", prob_filename.c_str());
        return false;
      }
    }

    const std::string camera_info_filename = m_input_image_prefix + "camera_info_" + image_number_str + ".txt";
    ROS_INFO("generate_virtual_views: loading camera info: %s", camera_info_filename.c_str());
    sensor_msgs::CameraInfo camera_info;
    if (!message_serialization::deserialize(camera_info_filename, camera_info))
    {
      ROS_ERROR("generate_virtual_views: could not open or deserialize file!");
      return false;
    }

    Eigen::Affine3f pose;
    {
      const std::string pose_filename = m_input_image_prefix + "pose_" + image_number_str + ".matrix";
      ROS_INFO("generate_virtual_views: loading pose from %s", pose_filename.c_str());
      if (!LoadPose(pose_filename, pose))
        return false;
      ROS_INFO_STREAM("generate_virtual_views: loaded pose " << pose.matrix() << "\n");
    }

    nbv_3d_cnn_evaluate_view_msgs::GroundTruthEvaluateViewGoal ev_goal;
    ev_goal.center_x = camera_info.K[2];
    ev_goal.center_y = camera_info.K[5];
    ev_goal.focal_x = camera_info.K[0];
    ev_goal.focal_y = camera_info.K[4];
    ev_goal.size_x = camera_info.width;
    ev_goal.size_y = camera_info.height;

    tf::poseEigenToMsg(pose.cast<double>(), ev_goal.pose);

    {
      cv_bridge::CvImagePtr cv_bridge(new cv_bridge::CvImage);
      max_depth.convertTo(max_depth, CV_32FC1, 1.0f / 1000.0f);
      cv_bridge->image = max_depth;
      cv_bridge->encoding = "32FC1";
      ev_goal.depth_image = *cv_bridge->toImageMsg();
    }

    {
      std::ifstream ifile(m_poi_file_name);
      float px, py, pz, pr;
      ifile >> px >> py >> pz >> pr;
      if (!ifile)
      {
        ROS_FATAL("generate_virtual_views: could not load poi from file %s", m_poi_file_name.c_str());
        exit(1);
      }
      ev_goal.has_roi_sphere = true;
      ev_goal.roi_sphere_center.x = px;
      ev_goal.roi_sphere_center.y = py;
      ev_goal.roi_sphere_center.z = pz;
      ev_goal.roi_sphere_radius = pr;
    }

    m_ground_truth_evaluate_view_ac->sendGoal(ev_goal);
    m_ground_truth_evaluate_view_ac->waitForResult();

    nbv_3d_cnn_evaluate_view_msgs::GroundTruthEvaluateViewResultConstPtr result =
        m_ground_truth_evaluate_view_ac->getResult();

    {
      cv_bridge::CvImagePtr cv_bridge = cv_bridge::toCvCopy(result->predicted_score_image);
      cv::Mat scores = cv_bridge->image;
      scores.convertTo(scores, CV_16UC1, 10.0 * 1000.0);
      const std::string scores_filename = m_output_image_prefix + "gt_scores_" + environment_number_str +
                                          "_" + image_number_str + ".png";
      ROS_INFO("generate_virtual_views: saving gt scores file %s", scores_filename.c_str());
      if (!cv::imwrite(scores_filename, scores))
      {
        ROS_ERROR("generate_virtual_views: could not save file %s", scores_filename.c_str());
        return false;
      }
    }

    {
      const std::string scores_filename = m_output_image_prefix + "gt_score.txt";
      std::ofstream ofile(scores_filename.c_str(), std::ios_base::app | std::ios_base::out);
      ofile << number << "\t" << environment_number << "\t" << result->gain << "\n";
    }

    return true;
  }

  bool Run()
  {
    if (m_image_number_i >= m_image_number_count)
    {
      ROS_INFO("generate_virtual_views: all image processed, terminated.");
      return false;
    }

    const uint32 image_number = m_image_number + m_image_number_i;
    ROS_INFO("generate_virtual_views: processing voxelgrid %u image %u.", unsigned(m_voxelgrid_i), unsigned(image_number));

    if (m_image_number_i == 0)
    {
      const std::string voxelgrid_filename = m_voxelgrid_prefix + "partial_" + std::to_string(m_voxelgrid_i) + ".voxelgrid";
      LoadVoxelgrid(voxelgrid_filename, m_voxelgrid_metadata_filename, m_tsdf_volume_filename);
    }

    cv::Mat predicted_depth_image, predicted_normal_image, predicted_robot_image, predicted_robot_normal_image;
    {
      const bool ok = RawProjection(image_number, predicted_depth_image,
                                    predicted_normal_image, predicted_robot_image,
                                    predicted_robot_normal_image);

      if (!ok)
      {
        ROS_WARN("generate_virtual_views: load failed, trying next voxelgrid.");
        m_image_number_i = 0;
        m_voxelgrid_i++;
        return true;
      }
    }

    if (m_image_number_i == 0 && m_voxelgrid_i == 0)
    {
      SaveCircleRectangleMask(predicted_depth_image.cols, predicted_depth_image.rows);
    }

    {
      const bool ok = GetGroundTruth(image_number, m_voxelgrid_i);
      if (!ok)
      {
        ROS_WARN("generate_virtual_views: load failed, terminated.");
        m_timer.stop();
        return false;
      }
    }

    const std::string image_number_str = std::to_string(image_number);
    const std::string voxelgrid_number_str = std::to_string(m_voxelgrid_i);

    {
      const std::string predicted_depth_filename = m_output_image_prefix + "raw_depth_" +
        voxelgrid_number_str + "_" + image_number_str + ".png";
      ROS_INFO("generate_virtual_views: saving file %s", predicted_depth_filename.c_str());
      cv::Mat predicted_depth_image_mult;
      predicted_depth_image.convertTo(predicted_depth_image_mult, CV_16UC1, 1000.0f);
      cv::imwrite(predicted_depth_filename, predicted_depth_image_mult);
    }

    {
      const std::string predicted_normal_filename = m_output_image_prefix + "raw_normal_" +
        voxelgrid_number_str + "_" + image_number_str + ".png";
      ROS_INFO("generate_virtual_views: saving file %s", predicted_normal_filename.c_str());
      cv::Mat predicted_normal_image_mult;
      predicted_normal_image.convertTo(predicted_normal_image_mult, CV_8UC3, 127.0, 127.0);
      cv::imwrite(predicted_normal_filename, predicted_normal_image_mult);
    }

    {
      const std::string predicted_depth_filename = m_output_image_prefix + "raw_robot_" +
          voxelgrid_number_str + "_"  + image_number_str + ".png";
      ROS_INFO("generate_virtual_views: saving file %s", predicted_depth_filename.c_str());
      cv::Mat predicted_depth_image_mult;
      predicted_robot_image.convertTo(predicted_depth_image_mult, CV_16UC1, 1000.0f);
      cv::imwrite(predicted_depth_filename, predicted_depth_image_mult);
    }

    {
      const std::string predicted_normal_filename = m_output_image_prefix + "raw_robot_normal_" +
          voxelgrid_number_str + "_"  + image_number_str + ".png";
      ROS_INFO("generate_virtual_views: saving file %s", predicted_normal_filename.c_str());
      cv::Mat predicted_normal_image_mult;
      predicted_robot_normal_image.convertTo(predicted_normal_image_mult, CV_8UC3, 127.0, 127.0);
      cv::imwrite(predicted_normal_filename, predicted_normal_image_mult);
    }

    m_image_number_i++;

    return true;
  }

  void onTimer(const ros::TimerEvent &)
  {
    if (!Run())
      return;

    m_timer.stop();
    m_timer.setPeriod(ros::Duration(0.0));
    m_timer.start();
  }

  private:
  ros::NodeHandle & m_nh;

  std::string m_voxelgrid_prefix;
  std::string m_tsdf_volume_filename;
  std::string m_voxelgrid_metadata_filename;
  std::string m_input_image_prefix;
  std::string m_output_image_prefix;
  uint32 m_image_number;
  uint32 m_image_number_count;
  uint32 m_depth_images_per_pose;

  uint32 m_image_number_i;
  uint32 m_voxelgrid_i;

  std::string m_poi_file_name;

  float m_max_range;
  float m_min_range;

  SetEnvironmentActionClientPtr m_set_environment_ac;
  RawProjectionActionClientPtr m_raw_projection_ac;
  GenerateFilterCircleRectangleMaskActionClientPtr m_filter_circle_rectangle_mask_ac;
  GroundTruthEvaluateViewActionClientPtr m_ground_truth_evaluate_view_ac;

  float m_cnn_matrix_voxel_size;
  Eigen::Vector3i m_cnn_matrix_size;
  Eigen::Vector3f m_cnn_matrix_origin;

  ros::Timer m_timer;
};

int main(int argc, char ** argv)
{
  ros::init(argc, argv, "generate_virtual_views");

  ros::NodeHandle nh("~");

  GenerateVirtualViews gvv(nh);

  ros::spin();

  return 0;
}
