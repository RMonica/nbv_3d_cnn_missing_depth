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

#include "cnn_real_image_projection.h"

#include <ros/ros.h>
#include <actionlib/client/simple_action_client.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Image.h>
#include <pcl_conversions/pcl_conversions.h>
#include <eigen_conversions/eigen_msg.h>
#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/PoseStamped.h>

#include <Eigen/Dense>
#include <Eigen/StdVector>

#include <vector>
#include <string>
#include <memory>
#include <stdint.h>
#include <fstream>

#include <rmonica_voxelgrid_common/voxelgrid.h>
#include <nbv_3d_cnn_real_image_msgs/ProjectionPredictAction.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#define RESOLUTION 64
#define VOXEL_SIZE 0.011713

#define FRAME_ID "map"

class CNNRealImageProjection
{
  public:
  typedef rmonica_voxelgrid_common::Voxelgrid Voxelgrid;

  typedef uint64_t uint64;

  typedef actionlib::SimpleActionClient<nbv_3d_cnn_real_image_msgs::ProjectionPredictAction> ProjectionPredictActionClient;
  typedef std::shared_ptr<ProjectionPredictActionClient> ProjectionPredictActionClientPtr;

  typedef pcl::PointCloud<pcl::PointXYZI> PointXYZICloud;

  Eigen::Affine3f LoadAffine(const std::string & filename)
  {
    ROS_INFO("cnn_real_image_projection: LoadAffine: loading file %s", filename.c_str());
    std::ifstream ifile(filename.c_str());
    Eigen::Affine3f result;
    for (uint64 y = 0; y < 4; y++)
      for (uint64 x = 0; x < 4; x++)
      {
        float v;
        ifile >> v;
        result.matrix()(y, x) = v;
      }
    if (!ifile)
    {
      ROS_FATAL("cnn_real_image_projection: LoadAffine: could not load file %s", filename.c_str());
      exit(1);
    }

    return result;
  }

  CNNRealImageProjection(ros::NodeHandle & nh): m_nh(nh)
  {
    std::string param_string;
    m_nh.param<std::string>("projection_predict_action", param_string, "");
    m_projection_predict_ac.reset(new ProjectionPredictActionClient(param_string, true));

    m_nh.param<std::string>("environment_filename", m_environment_filename, "");
    m_nh.param<std::string>("pose_filename", m_pose_filename, "");

    m_nh.param<std::string>("pose_topic", param_string, "camera_pose");
    m_camera_pose_pub = m_nh.advertise<geometry_msgs::PoseStamped>(param_string, 1);

    m_sent_environment_pub = m_nh.advertise<sensor_msgs::PointCloud2>("sent_environment", 1);

    m_recv_environment_pub = m_nh.advertise<sensor_msgs::PointCloud2>("recv_environment", 1);
    m_recv_image_pub = m_nh.advertise<sensor_msgs::Image>("recv_image", 1);

    m_timer = m_nh.createTimer(ros::Duration(0.0), &CNNRealImageProjection::Run, this, true);
  }

  void Run(const ros::TimerEvent&)
  {
    nbv_3d_cnn_real_image_msgs::ProjectionPredictGoal goal;

    PointXYZICloud sent_cloud;
    Voxelgrid environment(Eigen::Vector3i::Ones() * RESOLUTION);

    Eigen::Vector3f environment_origin(-float(RESOLUTION) * VOXEL_SIZE / 2,
                                       -float(RESOLUTION) * VOXEL_SIZE / 2,
                                       -float(RESOLUTION) * VOXEL_SIZE / 2);

    if (!m_environment_filename.empty())
    {
      ROS_INFO("cnn_real_image_projection: loading environent voxelgrid %s", m_environment_filename.c_str());
      Voxelgrid::Ptr vg = Voxelgrid::FromFileTernaryBinary(m_environment_filename);
      if (!vg)
      {
        ROS_FATAL("cnn_real_image_projection: could not load environent voxelgrid %s", m_environment_filename.c_str());
        exit(1);
      }
      environment = *vg;

      environment_origin = Eigen::Vector3f(-0.200731, 0.299269, 0.197952);
    }
    else
    {
      for (uint64 z = 0; z < RESOLUTION; z++)
        for (uint64 y = 0; y < RESOLUTION; y++)
          for (uint64 x = 0; x < RESOLUTION; x++)
          {
            const int radius = RESOLUTION / 4;
            const Eigen::Vector3i center = Eigen::Vector3i::Ones() * RESOLUTION / 2;
            const Eigen::Vector3i pt(x, y, z);

            const bool in_sphere = (pt - center).norm() < radius;

            if (in_sphere)
              environment.at(x, y, z) = 1.0f;
            else
              environment.at(x, y, z) = -1.0f;
          }
    }

    const uint64 width = environment.GetWidth();
    const uint64 height = environment.GetHeight();
    const uint64 depth = environment.GetDepth();

    for (uint64 z = 0; z < depth; z++)
      for (uint64 y = 0; y < height; y++)
        for (uint64 x = 0; x < width; x++)
        {
          const float e = environment.at(x, y, z);
          if (e > -0.5f)
          {
            const Eigen::Vector3i pt(x, y, z);
            const Eigen::Vector3f fpt = pt.cast<float>() * VOXEL_SIZE + environment_origin;
            pcl::PointXYZI ppt;
            ppt.x = fpt.x();
            ppt.y = fpt.y();
            ppt.z = fpt.z();
            ppt.intensity = e;
            sent_cloud.push_back(ppt);
          }
        }

    sent_cloud.width = sent_cloud.size();
    sent_cloud.height = 1;
    sent_cloud.is_dense = true;

    Eigen::Affine3f pose = Eigen::Affine3f::Identity();
    if (m_pose_filename.empty())
    {
      pose.translation().z() = -1.0f;
      pose = Eigen::AngleAxisf(M_PI, Eigen::Vector3f::UnitX()) * pose;
      pose = Eigen::AngleAxisf(M_PI/4.0f, Eigen::Vector3f::UnitY()) * pose;
    }
    else
    {
      pose = LoadAffine(m_pose_filename);
    }
    ROS_INFO_STREAM("cnn_real_image_projection: loaded pose: \n" << pose.matrix() << "\n");
    tf::poseEigenToMsg(pose.cast<double>(), goal.camera_pose);

    goal.environment_width = environment.GetWidth();
    goal.environment_height = environment.GetHeight();
    goal.environment_depth = environment.GetDepth();
    goal.environment_voxel_size = VOXEL_SIZE;

    tf::pointEigenToMsg(environment_origin.cast<double>(), goal.environment_origin);

    goal.ternary_voxelgrid = environment.ToFloatVector();

    const uint64 SCALE = 1;
    const float fx = 517.177 / SCALE;
    const float fy = 518.680 / SCALE;
    const float cx = 313.199 / SCALE;
    const float cy = 238.733 / SCALE;

    goal.image_width = 640 / SCALE;
    goal.image_height = 480 / SCALE;
    goal.image_depth = 64;
    goal.max_range = 1.5;
    goal.fx = fx;
    goal.fy = fy;
    goal.cx = cx;
    goal.cy = cy;

    goal.output_mask.resize(goal.image_width * goal.image_height, 1.0f);

    cv_bridge::CvImagePtr cv_bridge(new cv_bridge::CvImage);

    cv::Mat depth_image = cv::Mat(goal.image_height, goal.image_width, CV_32FC1, 0.0f);
    cv::Mat robot_image = cv::Mat(goal.image_height, goal.image_width, CV_32FC1, 0.0f);
    cv::Mat normal_image = cv::Mat(goal.image_height, goal.image_width, CV_32FC3, 0.0f);
    cv::Mat robot_normal_image = cv::Mat(goal.image_height, goal.image_width, CV_32FC3, 0.0f);

    cv_bridge->image = depth_image;
    cv_bridge->encoding = "32FC1";
    goal.depth_image = *cv_bridge->toImageMsg();

    cv_bridge->image = robot_image;
    cv_bridge->encoding = "32FC1";
    goal.robot_image = *cv_bridge->toImageMsg();

    cv_bridge->image = normal_image;
    cv_bridge->encoding = "32FC3";
    goal.normal_image = *cv_bridge->toImageMsg();

    cv_bridge->image = robot_normal_image;
    cv_bridge->encoding = "32FC3";
    goal.robot_normal_image = *cv_bridge->toImageMsg();

    ROS_INFO("cnn_real_image_prediction: waiting for server...");
    m_projection_predict_ac->waitForServer();

    ROS_INFO("cnn_real_image_prediction: sending goal...");

    m_projection_predict_ac->sendGoal(goal);

    ROS_INFO("cnn_real_image_prediction: waiting for result...");
    m_projection_predict_ac->waitForResult();

    nbv_3d_cnn_real_image_msgs::ProjectionPredictResult result = *m_projection_predict_ac->getResult();
    ROS_INFO("cnn_real_image_prediction: got result.");

    ROS_INFO("cnn_real_image_prediction: got image with shape %u %u", unsigned(result.image_height),
             unsigned(result.image_width));

    cv_bridge = cv_bridge::toCvCopy(result.probability_mask);
    cv::Mat result_image = cv_bridge->image;

    ROS_INFO("cnn_real_image_prediction: data length is %ux%u", unsigned(result_image.rows), unsigned(result_image.cols));

    {
      cv_bridge->image = result_image;
      cv_bridge->encoding = "32FC1";
      m_recv_image_pub.publish(cv_bridge->toImageMsg());
    }

//    PointXYZICloud received_cloud;
//    if (!result.occupancy_prob.empty())
//    {
//      for (uint64 zi = 0; zi < depth; zi++)
//        for (uint64 yi = 0; yi < height; yi++)
//          for (uint64 xi = 0; xi < width; xi++)

//          {

//            const uint64 i = zi * width * height + xi + yi * width;
//            const float intensity = result.occupancy_prob[i];

//            pcl::PointXYZI pt;
//            pt.x = xi * VOXEL_SIZE;
//            pt.y = yi * VOXEL_SIZE;
//            pt.z = zi * VOXEL_SIZE;
//            pt.intensity = intensity;
//            if (intensity > 0.5)
//              received_cloud.push_back(pt);
//          }
//      received_cloud.width = received_cloud.size();
//      received_cloud.height = 1;
//      received_cloud.is_dense = true;

//      {
//        sensor_msgs::PointCloud2 recv_cloud_msg;
//        pcl::toROSMsg(received_cloud, recv_cloud_msg);
//        recv_cloud_msg.header.frame_id = "map";
//        m_recv_environment_pub.publish(recv_cloud_msg);
//      }
//    }

    {
      sensor_msgs::PointCloud2 sent_cloud_msg;
      pcl::toROSMsg(sent_cloud, sent_cloud_msg);
      sent_cloud_msg.header.frame_id = FRAME_ID;
      m_sent_environment_pub.publish(sent_cloud_msg);
    }

    {
      geometry_msgs::PoseStamped pose_stamped;
      pose_stamped.header.frame_id = FRAME_ID;
      pose_stamped.header.stamp = ros::Time::now();
      tf::poseEigenToMsg(pose.cast<double>(), pose_stamped.pose);
      m_camera_pose_pub.publish(pose_stamped);
    }
  }

  private:
  ros::NodeHandle & m_nh;

  ProjectionPredictActionClientPtr m_projection_predict_ac;

  ros::Publisher m_sent_environment_pub;
  ros::Publisher m_recv_environment_pub;
  ros::Publisher m_recv_image_pub;
  ros::Publisher m_camera_pose_pub;

  std::string m_environment_filename;
  std::string m_pose_filename;

  ros::Timer m_timer;
};

int main(int argc, char ** argv)
{
  ros::init(argc, argv, "cnn_real_image_prediction");

  ros::NodeHandle nh("~");
  CNNRealImageProjection crip(nh);

  ros::spin();

  return 0;
}
