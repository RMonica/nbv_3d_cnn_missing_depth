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
