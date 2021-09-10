#include "render_robot_urdf.h"

// ROS
#include <ros/ros.h>
#include <eigen_conversions/eigen_msg.h>
#include <cv_bridge/cv_bridge.h>

// Eigen
#include <Eigen/Dense>
#include <Eigen/StdVector>

// custom
#include <render_robot_urdf/RenderRobotUrdf.h>
#include <init_fake_opengl_context/fake_opengl_context.h>

// OpenCV
#include <opencv2/core/mat.hpp>

// STL
#include <vector>
#include <memory>
#include <string>
#include <stdint.h>

class RenderRobotURDFNode
{
  public:
  typedef std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f> > Vector3fVector;
  typedef uint64_t uint64;
  typedef std::vector<float> FloatVector;

  typedef std::shared_ptr<RenderRobotURDF> RenderRobotURDFPtr;

  RenderRobotURDFNode(ros::NodeHandle & nh): m_nh(nh)
  {
    std::string param_string;

    m_nh.param<std::string>("service_name", param_string, "render_robot_urdf");
    m_service_server = m_nh.advertiseService("render_robot_urdf", &RenderRobotURDFNode::onService, this);

    m_nh.param<bool>("with_color", m_with_color, true);
    m_nh.param<bool>("with_normals", m_with_normals, true);

    m_nh.param<std::string>("model_param_name", m_model_param_name, "");

    m_rru.reset(new RenderRobotURDF(m_nh, m_with_color, m_with_normals, m_model_param_name));
  }

  bool onService(render_robot_urdf::RenderRobotUrdf::Request & request,
                 render_robot_urdf::RenderRobotUrdf::Response & response)
  {
    ROS_INFO("RenderRobotURDFNode: onService");

    const uint64 width = request.width;
    const uint64 height = request.height;
    const Eigen::Vector2f center(request.center_x, request.center_y);
    const Eigen::Vector2f focal(request.focal_x, request.focal_y);
    const Eigen::Vector2f range(request.min_range, request.max_range);
    const Eigen::Vector2i size(request.width, request.height);

    m_rru->SetIntrinsics(size, center, focal, range);

    Eigen::Affine3d camera_pose;
    tf::poseMsgToEigen(request.camera_pose, camera_pose);
    Eigen::Affine3d robot_pose;
    tf::poseMsgToEigen(request.robot_pose, robot_pose);

    const sensor_msgs::JointState & joint_state = request.joint_state;

    m_rru->Render(camera_pose, &joint_state, robot_pose);

    if (m_with_color)
    {
      Vector3fVector colors = m_rru->GetColorResult();

      cv::Mat colors_mat = cv::Mat(size.y(), size.x(), CV_32FC3);
      for (uint64 y = 0; y < height; y++)
        for (uint64 x = 0; x < width; x++)
        {
          for (uint64 c = 0; c < 3; c++)
            colors_mat.at<cv::Vec3f>(y, x)[c] = colors[y * width + x][c];
        }

      {
        cv_bridge::CvImage cv_bridge;
        cv_bridge.image = colors_mat;
        cv_bridge.encoding = "32FC3";
        response.color_image = *cv_bridge.toImageMsg();
      }
    }

    {
      FloatVector depths = m_rru->GetDepthResult();
      cv::Mat depths_mat = cv::Mat(size.y(), size.x(), CV_32FC1);

      for (uint64 y = 0; y < height; y++)
        for (uint64 x = 0; x < width; x++)
        {
          depths_mat.at<float>(y, x) = depths[y * width + x];
        }

      {
        cv_bridge::CvImage cv_bridge;
        cv_bridge.image = depths_mat;
        cv_bridge.encoding = "32FC1";
        response.depth_image = *cv_bridge.toImageMsg();
      }
    }

    if (m_with_normals)
    {
      Vector3fVector normals = m_rru->GetNormalResult();

      cv::Mat normals_mat = cv::Mat(size.y(), size.x(), CV_32FC3);
      for (uint64 y = 0; y < height; y++)
        for (uint64 x = 0; x < width; x++)
        {
          for (uint64 c = 0; c < 3; c++)
            normals_mat.at<cv::Vec3f>(y, x)[c] = normals[y * width + x][c];
        }

      {
        cv_bridge::CvImage cv_bridge;
        cv_bridge.image = normals_mat;
        cv_bridge.encoding = "32FC3";
        response.normal_image = *cv_bridge.toImageMsg();
      }
    }

    return true;
  }


  private:
  bool m_with_color;
  bool m_with_normals;

  std::string m_model_param_name;

  ros::NodeHandle & m_nh;

  RenderRobotURDFPtr m_rru;

  ros::ServiceServer m_service_server;
};

int main(int argc, char ** argv)
{
  ros::init(argc, argv, "render_robot_urdf");
  ros::NodeHandle nh("~");

  std::string screen;
  nh.param<std::string>("opengl_screen_name", screen, "");
  InitFakeOpenGLContext(screen);

  RenderRobotURDFNode rru(nh);

  ros::spin();

  return 0;
}
