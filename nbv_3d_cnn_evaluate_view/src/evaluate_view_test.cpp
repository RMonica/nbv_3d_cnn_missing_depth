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

#include "evaluate_view_test.h"

// ROS
#include <ros/ros.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <actionlib/client/simple_action_client.h>
#include <nbv_3d_cnn_evaluate_view_msgs/SetEnvironmentAction.h>
#include <nbv_3d_cnn_evaluate_view_msgs/EvaluateViewAction.h>
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

class TestEvaluateView
{
  public:
  typedef rmonica_voxelgrid_common::Voxelgrid Voxelgrid;
  typedef VoxelgridMetadata::Metadata Metadata;
  typedef VoxelgridMetadata::MetadataPtr MetadataPtr;

  typedef actionlib::SimpleActionClient<nbv_3d_cnn_evaluate_view_msgs::SetEnvironmentAction> SetEnvironmentActionClient;
  typedef std::shared_ptr<SetEnvironmentActionClient> SetEnvironmentActionClientPtr;
  typedef actionlib::SimpleActionClient<nbv_3d_cnn_evaluate_view_msgs::EvaluateViewAction> EvaluateViewActionClient;
  typedef std::shared_ptr<EvaluateViewActionClient> EvaluateViewActionClientPtr;

  typedef std::vector<float> FloatVector;
  typedef std::vector<double> DoubleVector;

  typedef pcl::PointCloud<pcl::PointXYZI> PointXYZICloud;
  typedef pcl::PointCloud<pcl::PointXYZ> PointXYZCloud;

  typedef uint32_t uint32;
  typedef uint64_t uint64;
  typedef uint8_t uint8;

  enum
  {
    CLEAR_POI_REGION_NONE = 0,
    CLEAR_POI_REGION_YP   = 1,
    CLEAR_POI_REGION_YN   = 2,
    CLEAR_POI_REGION_ALL  = 3,
  };

  TestEvaluateView(ros::NodeHandle & nh): m_nh(nh)
  {
    std::string param_string;
    int param_int;
    double param_double;
    m_nh.param<std::string>("voxelgrid_filename", param_string, "");
    m_voxelgrid_filename = param_string;

    m_nh.param<std::string>("tsdf_volume_filename", param_string, "");
    m_tsdf_volume_filename = param_string;

    m_nh.param<std::string>("voxelgrid_metadata_filename", param_string, "");
    m_voxelgrid_metadata_filename = param_string;

    m_nh.param<std::string>("image_prefix", param_string, "");
    m_image_prefix = param_string;

    m_nh.param<int>("image_number", param_int, 0);
    m_image_number = param_int;

    m_nh.param<int>("image_number_count", param_int, 1);
    m_image_number_count = param_int;

    m_nh.param<double>("max_range", param_double, 4.0);
    m_max_range = param_double;

    m_nh.param<double>("eval_dist_threshold", param_double, 0.05);
    m_eval_dist_threshold = param_double;

    m_nh.param<double>("poi_filter_radius", param_double, 0.5);
    m_poi_filter_radius = param_double;

    m_nh.param<int>("clear_poi_region", param_int, CLEAR_POI_REGION_NONE);
    m_clear_poi_region = param_int;

    m_nh.param<bool>("mode_advanced", m_mode_advanced, true);

    m_nh.param<std::string>("poi", param_string, "0.3 0.8 0.4");
    {
      std::istringstream istr(param_string);
      istr >> m_poi.x() >> m_poi.y() >> m_poi.z();
      if (!istr)
      {
        ROS_ERROR("evaluate_view_test: could not parse POI string: \"%s\"", param_string.c_str());
        m_poi = Eigen::Vector3f::Zero();
      }
    }

    m_nh.param<std::string>("cnn_matrix_origin", param_string, "-0.8 -0.8 0.0");
    {
      std::istringstream istr(param_string);
      istr >> m_cnn_matrix_origin.x() >> m_cnn_matrix_origin.y() >> m_cnn_matrix_origin.z();
      if (!istr)
      {
        ROS_FATAL("evaluate_view_test: unable to parse bounding box min: %s", param_string.c_str());
        exit(1);
      }
    }

    m_nh.param<std::string>("cnn_matrix_size", param_string, "128 128 96");
    {
      std::istringstream istr(param_string);
      istr >> m_cnn_matrix_size.x() >> m_cnn_matrix_size.y() >> m_cnn_matrix_size.z();
      if (!istr)
      {
        ROS_FATAL("evaluate_view_test: unable to parse cnn matrix resolution: %s", param_string.c_str());
        exit(1);
      }
    }

    m_nh.param<double>("kinfu_voxel_size", param_double, double(0.005859375));
    m_kinfu_voxel_size = param_double;

    m_nh.param<double>("cnn_matrix_voxel_size", param_double, double(0.01171875));
    m_cnn_matrix_voxel_size = param_double;

    m_image_number_i = 0;

    m_view_publisher = m_nh.advertise<sensor_msgs::Image>("test_view", 1);
    m_gt_view_publisher = m_nh.advertise<sensor_msgs::Image>("test_gt_view", 1);
    m_cloud_publisher = m_nh.advertise<sensor_msgs::PointCloud2>("test_cloud", 1);
    m_gt_cloud_publisher = m_nh.advertise<sensor_msgs::PointCloud2>("test_gt_cloud", 1);
    m_comparison_view_publisher = m_nh.advertise<sensor_msgs::Image>("test_comp_view", 1);
    m_debug_normals_publisher = m_nh.advertise<sensor_msgs::Image>("debug_normals", 1);
    m_debug_depth_publisher = m_nh.advertise<sensor_msgs::Image>("debug_depth", 1);
    m_debug_scores_publisher = m_nh.advertise<sensor_msgs::Image>("debug_scores", 1);
    m_debug_voxelgrid_pub = m_nh.advertise<sensor_msgs::PointCloud2>("debug_voxelgrid", 1);

    m_set_environment_ac.reset(new SetEnvironmentActionClient("/evaluate_view/set_environment", true));
    m_evaluate_view_ac.reset(new EvaluateViewActionClient("/evaluate_view/evaluate_view", true));

    ROS_INFO("test_evaluate_view: waiting for server...");

    m_set_environment_ac->waitForServer();
    m_evaluate_view_ac->waitForServer();

    ROS_INFO("test_evaluate_view: done.");

    m_timer = m_nh.createTimer(ros::Duration(1.0), &TestEvaluateView::onTimer, this, true);
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

  PointXYZCloud DepthImageToCloud(const cv::Mat & image,
                                  const Eigen::Vector2f & focal,
                                  const Eigen::Vector2f & center)
  {
    PointXYZCloud result;

    const uint64 width = image.cols;
    const uint64 height = image.rows;
    result.reserve(width * height);

    for (uint64 y = 0; y < height; y++)
      for (uint64 x = 0; x < width; x++)
      {
        const float z = image.at<float>(y, x);
        const Eigen::Vector3f ept((x + 0.5 - center.x()) * z / focal.x(),
                                  (y + 0.5 - center.y()) * z / focal.y(),
                                  z
                                  );
        const pcl::PointXYZ pt(ept.x(), ept.y(), ept.z());
        result.push_back(pt);
      }
    result.width = width;
    result.height = height;
    result.is_dense = false;

    return result;
  }

  sensor_msgs::PointCloud2 CloudToMsg(const PointXYZCloud & cloud)
  {
    sensor_msgs::PointCloud2 result;
    pcl::toROSMsg(cloud, result);
    result.header.frame_id = "map";
    return result;
  }

  void DebugPublishVoxelgrid(ros::Publisher & publisher, const Voxelgrid & vg,
                             const float voxel_size, const Eigen::Vector3f & origin,
                             const Voxelgrid & prob_vg, const Eigen::Vector3i & prob_vg_origin)
  {
    const uint64 width = vg.GetWidth();
    const uint64 height = vg.GetHeight();
    const uint64 depth = vg.GetDepth();

    PointXYZICloud cloud;

    ROS_INFO_STREAM("poi is: " << m_poi.transpose());
    ROS_INFO_STREAM("origin is: " << origin.transpose());

    const Eigen::Vector3i prob_vg_size = prob_vg.GetSize();

    for (uint64 z = 0; z < depth; z++)
      for (uint64 y = 0; y < height; y++)
        for (uint64 x = 0; x < width; x++)
        {
          const Eigen::Vector3i v3(x, y, z);
          const Eigen::Vector3i prob_v3 = v3 - prob_vg_origin;

          float prob = -1.0;
          if ((prob_v3.array() >= 0).all() && (prob_v3.array() < prob_vg_size.array()).all())
          {
            prob = prob_vg.at(prob_v3);
          }

          const Eigen::Vector3f pt = v3.cast<float>() * voxel_size + origin;

          if (((pt - m_poi).array().abs() > m_poi_filter_radius).any())
            continue;

          if (prob >= 0.0)
          {
            if (prob < 0.1)
              continue;
          }

          float v = vg.at(v3);
          if (v < -0.5f)
            continue; // empty

          float intensity = 0.0;
          if (v > 0.5)
            intensity = 1.0;
          else
          {
            if (prob >= 0.0)
              intensity = prob;
          }

          pcl::PointXYZI ppt;
          ppt.x = pt.x();
          ppt.y = pt.y();
          ppt.z = pt.z();
          ppt.intensity = intensity;
          cloud.push_back(ppt);
        }

    sensor_msgs::PointCloud2 cloud_msg;
    pcl::toROSMsg(cloud, cloud_msg);
    cloud_msg.header.frame_id = "map";
    cloud_msg.header.stamp = ros::Time::now();
    publisher.publish(cloud_msg);
  }

  void LoadVoxelgrid(const std::string & voxelgrid_filename, const std::string & metadata_filename,
                     const std::string & tsdf_volume_filename)
  {
    ROS_INFO("test_evaluate_view: loading voxelgrid from file %s", voxelgrid_filename.c_str());
    const Voxelgrid::Ptr vg = Voxelgrid::FromFileTernaryBinary(voxelgrid_filename);
    if (!vg)
    {
      ROS_ERROR("test_evaluate_view: could not load voxelgrid.");
      exit(1);
    }

    const Eigen::Vector3i voxelgrid_size = vg->GetSize();

    ROS_INFO("test_evaluate_view: loading voxelgrid metadata from file %s", metadata_filename.c_str());
    MetadataPtr metadata = VoxelgridMetadata::LoadMetadata(metadata_filename);
    if (!metadata)
    {
      ROS_ERROR("test_evaluate_view: could not load grid metadata!");
      exit(1);
    }

    ROS_INFO("test_evaluate_view: loading TSDF from file %s", tsdf_volume_filename.c_str());
    PointXYZICloud tsdf_cloud;
    if (pcl::io::loadPCDFile(tsdf_volume_filename, tsdf_cloud) < 0)
    {
      ROS_ERROR("test_evaluate_view: could not load TSDF cloud!");
      exit(1);
    }
    ROS_INFO("test_evaluate_view: TSDF cloud has size %u", unsigned(tsdf_cloud.size()));

    const float voxel_size = metadata->voxel_size;

    Voxelgrid vg2 = *vg;
    if (m_clear_poi_region != CLEAR_POI_REGION_NONE)
    {
      for (uint64 z = 0; z < voxelgrid_size.z(); z++)
        for (uint64 y = 0; y < voxelgrid_size.y(); y++)
          for (uint64 x = 0; x < voxelgrid_size.x(); x++)
          {
            const Eigen::Vector3i v3(x, y, z);
            const Eigen::Vector3f pt = Eigen::Vector3f(v3.cast<float>().array() * voxel_size) + metadata->bbox_min;

            if ((pt - m_poi).norm() > m_poi_filter_radius)
              continue;

            if (m_clear_poi_region == CLEAR_POI_REGION_YN)
              if (pt.y() < m_poi.y())
                vg2.at(v3) = 0.0f;
            if (m_clear_poi_region == CLEAR_POI_REGION_YP)
              if (pt.y() > m_poi.y())
                vg2.at(v3) = 0.0f;
            if (m_clear_poi_region == CLEAR_POI_REGION_ALL)
              vg2.at(v3) = 0.0f;
          }
    }

    Voxelgrid prob_vg;
    Eigen::Vector3i probabilities_origin = Eigen::Vector3i::Zero();
    {
      nbv_3d_cnn_evaluate_view_msgs::SetEnvironmentGoal se_goal;
      se_goal.voxel_size = voxel_size;

      se_goal.environment_origin.x = metadata->bbox_min.x();
      se_goal.environment_origin.y = metadata->bbox_min.y();
      se_goal.environment_origin.z = metadata->bbox_min.z();

      se_goal.environment = vg2.ToInt8MultiArray();

      se_goal.probabilities_voxel_size = m_cnn_matrix_voxel_size;

      if (m_clear_poi_region == CLEAR_POI_REGION_NONE)
      {
        se_goal.probabilities_origin[0] = 0;
        se_goal.probabilities_origin[1] = 0;
        se_goal.probabilities_origin[2] = 0;
        se_goal.probabilities_size[0] = 0;
        se_goal.probabilities_size[1] = 0;
        se_goal.probabilities_size[2] = 0;
      }
      else
      {
        const float weird_probabilities_diameter = m_poi_filter_radius / 2.0f;
        probabilities_origin = ((m_cnn_matrix_origin - metadata->bbox_min) /
                                m_kinfu_voxel_size).cast<int>();

        const Eigen::Vector3i probabilities_size = m_cnn_matrix_size;

        se_goal.probabilities_origin[0] = probabilities_origin.x();
        se_goal.probabilities_origin[1] = probabilities_origin.y();
        se_goal.probabilities_origin[2] = probabilities_origin.z();
        se_goal.probabilities_size[0] = probabilities_size.x();
        se_goal.probabilities_size[1] = probabilities_size.y();
        se_goal.probabilities_size[2] = probabilities_size.z();
      }

      //const std::string temp_name = std::tmpnam(NULL);
      //vg2.ToFileTernaryBinary(temp_name);
      //se_goal.load_from_ternary_file = temp_name;

      //se_goal.environment_origin.y += voxel_size.y(); // FIXME: why?

      pcl::toROSMsg(tsdf_cloud, se_goal.tsdf_cloud);

      ROS_INFO("test_evaluate_view: setting environment...");
      m_set_environment_ac->sendGoal(se_goal);
      m_set_environment_ac->waitForResult();

      nbv_3d_cnn_evaluate_view_msgs::SetEnvironmentResultConstPtr se_result = m_set_environment_ac->getResult();
      prob_vg = *Voxelgrid::FromFloat32MultiArray(se_result->predicted_probabilities);

      //std::remove(temp_name.c_str());
    }

    DebugPublishVoxelgrid(m_debug_voxelgrid_pub, vg2, voxel_size, metadata->bbox_min,
                          prob_vg, probabilities_origin);

    ROS_INFO_STREAM("test_evaluate_view: bounding box is " << metadata->bbox_max.transpose() << " - "
                    << metadata->bbox_max.transpose());
    ROS_INFO_STREAM("test_evaluate_view: voxel size is " << voxel_size);
  }

  void EvaluateView(const uint64 number,
                    cv::Mat & gt_image,
                    cv::Mat & predicted_image)
  {
    cv_bridge::CvImagePtr cv_bridge(new cv_bridge::CvImage);

    ROS_INFO("test_evaluate_view: preparing evaluation...");

    const std::string image_number_str = std::to_string(number);

    const std::string camera_info_filename = m_image_prefix + "camera_info_" + image_number_str + ".txt";
    ROS_INFO("test_evaluate_view: loading camera info: %s", camera_info_filename.c_str());
    sensor_msgs::CameraInfo camera_info;
    if (!message_serialization::deserialize(camera_info_filename, camera_info))
    {
      ROS_ERROR("test_evaluate_view: could not open or deserialize file!");
      return;
    }

    const Eigen::Vector2f focal(camera_info.K[0], camera_info.K[4]);
    const Eigen::Vector2f center(camera_info.K[2], camera_info.K[5]);
    const Eigen::Vector2i image_size(camera_info.width, camera_info.height);

    //ROS_INFO_STREAM("test_evaluate_view: loaded camera info: " << camera_info);

    gt_image = cv::Mat(image_size.y(), image_size.x(), CV_32FC1, 0.0f);
    {
      cv::Mat gt_image_counter = cv::Mat(image_size.y(), image_size.x(), CV_32FC1, 0.0f);
      for (uint64 i = 0; i < 4; i++)
      {

        const std::string gt_image_filename = m_image_prefix + "depth_image_" + image_number_str + "_" +
            std::to_string(i) + ".png";

        ROS_INFO("test_evaluate_view: loading GT image: %s", gt_image_filename.c_str());

        cv::Mat gt_image_i = cv::imread(gt_image_filename, cv::IMREAD_ANYDEPTH);
        if (gt_image_i.empty())
        {
          ROS_ERROR("test_evaluate_view: could not load GT image!");
          return;
        }

        gt_image_i.convertTo(gt_image_i, CV_32FC1, 1.0f/1000.0f);

        for (uint64 y = 0; y < image_size.y(); y++)
          for (uint64 x = 0; x < image_size.x(); x++)
            if (gt_image_i.at<float>(y, x))
            {
              gt_image_counter.at<float>(y, x) += 1.0f;
              gt_image.at<float>(y, x) += gt_image_i.at<float>(y, x);
            }
      }

      for (uint64 y = 0; y < image_size.y(); y++)
        for (uint64 x = 0; x < image_size.x(); x++)
          if (gt_image_counter.at<float>(y, x))
            gt_image.at<float>(y, x) /= gt_image_counter.at<float>(y, x);
    }

    {
      cv_bridge->image = gt_image;
      cv_bridge->encoding = "32FC1";
      sensor_msgs::Image img_msg = *cv_bridge->toImageMsg();
      m_gt_view_publisher.publish(img_msg);

      const PointXYZCloud gt_cloud = DepthImageToCloud(gt_image, focal, center);
      m_gt_cloud_publisher.publish(CloudToMsg(gt_cloud));
    }

    Eigen::Affine3f pose;
    {
      const std::string pose_filename = m_image_prefix + "pose_" + image_number_str + ".matrix";
      ROS_INFO("test_evaluate_view: loading pose from %s", pose_filename.c_str());
      std::ifstream ifile(pose_filename.c_str());
      for (uint64 y = 0; y < 3; y++)
      {
        for (uint64 x = 0; x < 3; x++)
          ifile >> pose.linear()(y, x);
        ifile >> pose.translation()[y];
      }
      ROS_INFO_STREAM("test_evaluate_view: loaded pose " << pose.matrix() << "\n");
    }

    sensor_msgs::JointState joint_state;
    {
      const std::string joint_state_filename = m_image_prefix + "joint_state_" + image_number_str + ".txt";
      ROS_INFO("test_evaluate_view: loading joint state from %s", joint_state_filename.c_str());
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
        ROS_ERROR("test_evaluate_view: load joint state failed!");
        exit(1);
      }

      joint_state.position = js;
      joint_state.name = {"joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"};
    }

    nbv_3d_cnn_evaluate_view_msgs::EvaluateViewGoal ev_goal;
    ev_goal.center_x = camera_info.K[2];
    ev_goal.center_y = camera_info.K[5];
    ev_goal.focal_x = camera_info.K[0];
    ev_goal.focal_y = camera_info.K[4];
    ev_goal.size_x = camera_info.width;
    ev_goal.size_y = camera_info.height;

    ev_goal.joint_state = joint_state;

    ev_goal.max_range = 4.0;
    ev_goal.min_range = 0.4;

    ev_goal.has_roi_sphere = true;
    ev_goal.roi_sphere_center.x = m_poi.x();
    ev_goal.roi_sphere_center.y = m_poi.y();
    ev_goal.roi_sphere_center.z = m_poi.z();
    ev_goal.roi_sphere_radius = m_poi_filter_radius;

    if (m_mode_advanced)
      ev_goal.mode = ev_goal.MODE_ADVANCED_PROB;
    else
      ev_goal.mode = ev_goal.MODE_STANDARD_PROB;

    tf::poseEigenToMsg(pose.cast<double>(), ev_goal.pose);

    ROS_INFO("test_evaluate_view: sending evaluation goal...");
    m_evaluate_view_ac->sendGoal(ev_goal);
    m_evaluate_view_ac->waitForResult();

    nbv_3d_cnn_evaluate_view_msgs::EvaluateViewResultConstPtr ev_result = m_evaluate_view_ac->getResult();
    m_view_publisher.publish(ev_result->predicted_realistic_image);

    cv_bridge = cv_bridge::toCvCopy(ev_result->predicted_realistic_image);
    predicted_image = cv_bridge->image;
    const PointXYZCloud predicted_cloud = DepthImageToCloud(predicted_image, focal, center);
    m_cloud_publisher.publish(CloudToMsg(predicted_cloud));

    {
      cv_bridge = cv_bridge::toCvCopy(ev_result->predicted_depth_image);
      cv::Mat depth = cv_bridge->image;
      cv_bridge = cv_bridge::toCvCopy(ev_result->predicted_depth_image_status);
      cv::Mat status = cv_bridge->image;

      const uint64 width = depth.cols;
      const uint64 height = depth.rows;

      cv::Mat debug_depth(height, width, CV_8UC3, cv::Vec3b(0, 0, 0));
      for (uint64 y = 0; y < height; y++)
        for (uint64 x = 0; x < width; x++)
        {
          const float v = depth.at<float>(y, x);
          if (!v)
            continue;
          const float v_scaled = std::min((v / ev_goal.max_range) * 127.0, 127.0);
          const uint8 v_scaled_i = v_scaled;
          const uint8 s = status.at<uint8>(y, x);
          if (s == ev_result->STATUS_UNKNOWN)
            debug_depth.at<cv::Vec3b>(y, x) = cv::Vec3b(v_scaled_i, v_scaled_i, v_scaled_i);
          if (s == ev_result->STATUS_OCCUPIED)
            debug_depth.at<cv::Vec3b>(y, x) = cv::Vec3b(v_scaled_i + 127, v_scaled_i + 127, v_scaled_i + 127);
          if (s == ev_result->STATUS_LOST)
            debug_depth.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, v_scaled_i + 127);
        }
      cv_bridge->image = debug_depth;
      cv_bridge->encoding = "rgb8";
      m_debug_depth_publisher.publish(cv_bridge->toImageMsg());
    }

    {
      cv_bridge = cv_bridge::toCvCopy(ev_result->predicted_normal_image);
      cv_bridge->image.convertTo(cv_bridge->image, CV_8UC3, 127, 127);
      cv_bridge->encoding = "rgb8";
      m_debug_normals_publisher.publish(cv_bridge->toImageMsg());
    }

    {
      m_debug_scores_publisher.publish(ev_result->predicted_score_image);
    }
  }

  void onTimer(const ros::TimerEvent &)
  {
    if (m_image_number_i >= m_image_number_count)
    {
      ROS_INFO("test_evaluate_view: all image processed, terminated.");
      return;
    }

    const uint32 image_number = m_image_number + m_image_number_i;
    ROS_INFO("test_evaluate_view: processing image %u.", unsigned(image_number));

    if (m_image_number_i == 0)
    {
      LoadVoxelgrid(m_voxelgrid_filename, m_voxelgrid_metadata_filename, m_tsdf_volume_filename);
    }

    cv::Mat gt_image, predicted_image;
    EvaluateView(image_number, gt_image, predicted_image);

    uint64 fp, fn, tp, tn, dist_errors;
    cv::Mat eval_image = CompareImages(gt_image, predicted_image,
                                       fp, fn, tp, tn, dist_errors);

    const uint64 total = fp + fn + tp + tn + dist_errors;
    ROS_INFO("False positives: %u (%f%%)", unsigned(fp), float(fp) / float(total) * 100.0f);
    ROS_INFO("False negatives: %u (%f%%)", unsigned(fn), float(fn) / float(total) * 100.0f);
    ROS_INFO("True  positives: %u (%f%%)", unsigned(tp), float(tp) / float(total) * 100.0f);
    ROS_INFO("True  negatives: %u (%f%%)", unsigned(tn), float(tn) / float(total) * 100.0f);
    ROS_INFO("Distance errors: %u (%f%%)", unsigned(dist_errors), float(dist_errors) / float(total) * 100.0f);
    ROS_INFO("-- Total       : %u (%f%%)", unsigned(total), float(total) / float(total) * 100.0f);

    PublishCVImage(eval_image, m_comparison_view_publisher, "rgb8");

    m_image_number_i++;

    m_timer.stop();
    m_timer.setPeriod(ros::Duration(0.0));
    m_timer.start();
  }

  void PublishCVImage(const cv::Mat & image, ros::Publisher & pub, const std::string & encoding)
  {
    cv_bridge::CvImage cv_bridge;
    cv_bridge.image = image;
    cv_bridge.encoding = encoding;
    pub.publish(cv_bridge.toImageMsg());
  }

  cv::Mat CompareImages(const cv::Mat & ground_truth, const cv::Mat & predicted,
                        uint64 & fp, uint64 & fn, uint64 & tp, uint64 & tn, uint64 & dist_errors)
  {
    const uint64 width = ground_truth.cols;
    const uint64 height = ground_truth.rows;

    fp = fn = tp = tn = dist_errors = 0;

    cv::Mat result(height, width, CV_8UC3, cv::Vec3b(0, 0, 0));

    for (uint64 y = 0; y < height; y++)
      for (uint64 x = 0; x < width; x++)
      {
        if ((ground_truth.at<float>(y, x) == 0) &&
            (predicted.at<float>(y, x) != 0))
        {
          result.at<cv::Vec3b>(y, x) = cv::Vec3b(255, 0, 0);
          fp++;
        }
        if ((ground_truth.at<float>(y, x) != 0) &&
            (predicted.at<float>(y, x) == 0))
        {
          result.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 255);
          fn++;
        }
        if ((ground_truth.at<float>(y, x) == 0) &&
            (predicted.at<float>(y, x) == 0))
        {
          tn++;
        }
        if ((ground_truth.at<float>(y, x) != 0) &&
            (predicted.at<float>(y, x) != 0))
        {
          const float v1 = ground_truth.at<float>(y, x);
          const float v2 = predicted.at<float>(y, x);

          if (std::abs(v1 - v2) > m_eval_dist_threshold)
            dist_errors++;
          else
            tp++;
        }
      }

    return result;
  }

  private:
  ros::NodeHandle & m_nh;

  std::string m_voxelgrid_filename;
  std::string m_tsdf_volume_filename;
  std::string m_voxelgrid_metadata_filename;
  std::string m_image_prefix;
  uint32 m_image_number;
  uint32 m_image_number_count;

  Eigen::Vector3f m_poi;
  float m_poi_filter_radius;
  uint64 m_clear_poi_region;

  bool m_mode_advanced;

  uint32 m_image_number_i;

  float m_max_range;
  float m_eval_dist_threshold;

  ros::Publisher m_view_publisher;
  ros::Publisher m_cloud_publisher;
  ros::Publisher m_gt_view_publisher;
  ros::Publisher m_gt_cloud_publisher;
  ros::Publisher m_comparison_view_publisher;
  ros::Publisher m_debug_depth_publisher;
  ros::Publisher m_debug_normals_publisher;
  ros::Publisher m_debug_scores_publisher;
  ros::Publisher m_debug_voxelgrid_pub;

  SetEnvironmentActionClientPtr m_set_environment_ac;
  EvaluateViewActionClientPtr m_evaluate_view_ac;

  float m_kinfu_voxel_size;
  float m_cnn_matrix_voxel_size;
  Eigen::Vector3i m_cnn_matrix_size;
  Eigen::Vector3f m_cnn_matrix_origin;

  ros::Timer m_timer;
};

int main(int argc, char ** argv)
{
  ros::init(argc, argv, "test_evaluate_view");

  ros::NodeHandle nh("~");

  TestEvaluateView tev(nh);

  ros::spin();

  return 0;
}
