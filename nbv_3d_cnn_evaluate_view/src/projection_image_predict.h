#ifndef PROJECTION_IMAGE_PREDICT_H
#define PROJECTION_IMAGE_PREDICT_H

// custom
#include <rmonica_voxelgrid_common/voxelgrid.h>
#include <rmonica_voxelgrid_common/metadata.h>

#include <nbv_3d_cnn_real_image_msgs/ProjectionPredictAction.h>

// ROS
#include <ros/ros.h>
#include <actionlib/client/simple_action_client.h>

// OpenCV
#include <opencv2/core/core.hpp>

// STL
#include <vector>
#include <memory>
#include <string>
#include <stdint.h>

// Eigen
#include <Eigen/Dense>
#include <Eigen/StdVector>

class ProjectionImagePredict
{
  public:
  typedef actionlib::SimpleActionClient<nbv_3d_cnn_real_image_msgs::ProjectionPredictAction> ProjectionPredictActionClient;
  typedef std::shared_ptr<ProjectionPredictActionClient> ProjectionPredictActionClientPtr;
  typedef decltype(nbv_3d_cnn_real_image_msgs::ProjectionPredictResult().saliency_images) SaliencyImageVector;

  typedef rmonica_voxelgrid_common::Voxelgrid Voxelgrid;
  typedef VoxelgridMetadata::Metadata Metadata;

  typedef std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f> > Vector3fVector;

  typedef uint64_t uint64;

  ProjectionImagePredict(ros::NodeHandle & nh);

  bool Predict(const uint64 image_width,
               const uint64 image_height,
               const uint64 image_depth,
               const float fx,
               const float fy,
               const float cx,
               const float cy,
               const float max_range,
               const bool with_saliency_images,

               const Eigen::Affine3f & pose,

               const cv::Mat & depth_image,
               const cv::Mat & normal_image,
               const cv::Mat & robot_image,
               const cv::Mat & robot_normal_image,
               const cv::Mat & output_mask,

               cv::Mat & probability_mask);

  const SaliencyImageVector & GetLastSaliencyImages() const {return m_last_saliency_images; }

  double GetLastPredictionTime() const {return m_last_prediction_time; }

  void SetEnvironment(const Voxelgrid & environment,
                      const Eigen::Vector3f & environment_origin,
                      const float voxel_size);

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  private:

  ros::NodeHandle & m_nh;

  Voxelgrid::Ptr m_environment;
  float m_environment_voxel_size;
  Eigen::Vector3f m_environment_origin;

  double m_last_prediction_time;

  Eigen::Vector3f m_crop_bbox_min;
  Eigen::Vector3f m_crop_bbox_max;

  SaliencyImageVector m_last_saliency_images;

  ProjectionPredictActionClientPtr m_projection_predict_action_client;
};

#endif // PROJECTION_IMAGE_PREDICT_H
