// ROS
#include <ros/ros.h>
#include <eigen_conversions/eigen_msg.h>
#include <geometry_msgs/Pose.h>
#include <sensor_msgs/JointState.h>
#include <sensor_msgs/image_encodings.h>
#include <tf2_ros/transform_broadcaster.h>

#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>

// Eigen
#include <Eigen/Dense>
#include <Eigen/StdVector>

//STL
#include <vector>

#include <render_robot_urdf/RenderRobotUrdf.h>

int main(int argc,char ** argv)
{
  ros::init(argc,argv,"render_robot_urdf_test");
  ros::NodeHandle nh("~");

  tf2_ros::TransformBroadcaster tf_broadcaster;

  ros::Publisher color_image_pub = nh.advertise<sensor_msgs::Image>("color_image", 1);
  ros::Publisher depth_image_pub = nh.advertise<sensor_msgs::Image>("depth_image", 1);
  ros::Publisher normal_image_pub = nh.advertise<sensor_msgs::Image>("normal_image", 1);

  ros::ServiceClient service_client = nh.serviceClient<render_robot_urdf::RenderRobotUrdf>(
        "/render_robot_urdf/render_robot_urdf");

  ROS_INFO("render_robot_urdf_test: waiting 1 second...");
  ros::Duration(1.0).sleep();

  ROS_INFO("render_robot_urdf_test: waiting for existence...");
  service_client.waitForExistence();

  ROS_INFO("render_robot_urdf_test: sending service call...");

  Eigen::Matrix4d camera_pose_matrix;
//  camera_pose_matrix <<  -0.206334,   0.862385,  -0.462296,   0.624177,
//                          0.841652, -0.0845435,  -0.533361,    1.17406,
//                         -0.499047,  -0.499143,  -0.708384,   0.895559,
//                                 0,          0,         0,          1;
  camera_pose_matrix <<  -0.254921,   0.743371,   0.618397,   -0.13259,
                         0.284997,   0.668879,   -0.68657,    1.28064,
                        -0.924009, 0.00121985,  -0.382369,   0.668508,
                                0,          0,          0,         1;
  Eigen::Affine3d camera_pose;
  camera_pose.matrix() = camera_pose_matrix;
  Eigen::Affine3d robot_pose = Eigen::Affine3d::Identity();

  {
    geometry_msgs::TransformStamped transformStamped;

    transformStamped.header.stamp = ros::Time::now();
    transformStamped.header.frame_id = "map";
    transformStamped.child_frame_id = "test_pose";
    tf::transformEigenToMsg(camera_pose, transformStamped.transform);

    tf_broadcaster.sendTransform(transformStamped);
  }

  sensor_msgs::JointState joint_state;

  joint_state.name = {"joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"};
  //joint_state.position = {-1.03425, 0.881409, -0.518607, -0.802014, 1.01142, 0.289096};
  joint_state.position = {-1.711, 0.781942, -0.995972, -0.258344, 1.041, 0.640728 };

  render_robot_urdf::RenderRobotUrdf srv;
  srv.request.center_x = 313.1992494681865;
  srv.request.center_y = 238.7333517339323;
  srv.request.focal_x = 517.1771595680259;
  srv.request.focal_y = 518.6800755814886;
  srv.request.min_range = 0.1;
  srv.request.max_range = 4.0;
  srv.request.width = 640;
  srv.request.height = 480;

  tf::poseEigenToMsg(camera_pose, srv.request.camera_pose);
  tf::poseEigenToMsg(robot_pose, srv.request.robot_pose);

  srv.request.joint_state = joint_state;

  if (service_client.call(srv))
  {
    ROS_INFO("render_robot_urdf_test: call succeeded!");

    {
      cv_bridge::CvImage cv_bridge = *cv_bridge::toCvCopy(srv.response.color_image);
      cv_bridge.image.convertTo(cv_bridge.image, CV_8UC3, 255.0);
      cv_bridge.encoding = "rgb8";
      sensor_msgs::Image img_out = *cv_bridge.toImageMsg();

      color_image_pub.publish(img_out);
    }

    {
      cv_bridge::CvImage cv_bridge = *cv_bridge::toCvCopy(srv.response.depth_image);
      cv_bridge.encoding = "32FC1";
      sensor_msgs::Image img_out = *cv_bridge.toImageMsg();

      depth_image_pub.publish(img_out);
    }

    if (!srv.response.normal_image.data.size())
    {
      ROS_ERROR("render_robot_urdf_test: normal image is empty!");
    }
    else
    {
      cv_bridge::CvImage cv_bridge = *cv_bridge::toCvCopy(srv.response.normal_image);
      cv_bridge.image.convertTo(cv_bridge.image, CV_8UC3, 127.0, 127.0);
      cv_bridge.encoding = "rgb8";
      sensor_msgs::Image img_out = *cv_bridge.toImageMsg();

      normal_image_pub.publish(img_out);
    }
  }
  else
  {
    ROS_ERROR("render_robot_urdf_test: call failed!");
  }

  ros::spin();

  return 0;
}
