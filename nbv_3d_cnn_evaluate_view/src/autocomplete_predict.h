#ifndef AUTOCOMPLETE_PREDICT_H
#define AUTOCOMPLETE_PREDICT_H

// custom
#include <rmonica_voxelgrid_common/voxelgrid.h>
#include <nbv_3d_cnn_msgs/Floats.h>
#include <nbv_3d_cnn_msgs/Predict3dAction.h>

// ROS
#include <ros/ros.h>
#include <actionlib/client/simple_action_client.h>

// STL
#include <vector>
#include <memory>
#include <string>

class AutocompletePredict
{
  public:
  typedef rmonica_voxelgrid_common::Voxelgrid Voxelgrid;

  typedef actionlib::SimpleActionClient<nbv_3d_cnn_msgs::Predict3dAction> Predict3dActionClient;
  typedef std::shared_ptr<Predict3dActionClient> Predict3dActionClientPtr;

  AutocompletePredict(ros::NodeHandle & nh);

  bool Predict3d(const Voxelgrid & empty, const Voxelgrid & occupied, Voxelgrid & autocompleted);

  void DownsampleKinfuVoxelgrid(const Voxelgrid & kinfu_voxelgrid,
                                const float kinfu_voxel_size,
                                const float cnn_voxel_size,
                                const Eigen::Vector3i & kinfu_offset,
                                const Eigen::Vector3i & output_resolution,
                                Voxelgrid & cnn_matrix_empty,
                                Voxelgrid & cnn_matrix_occupied,
                                Voxelgrid & downsampled_voxelgrid);

  private:
  void onRawData(const nbv_3d_cnn_msgs::FloatsConstPtr raw_data);

  ros::NodeHandle & m_nh;

  ros::NodeHandle m_private_nh;
  Predict3dActionClientPtr m_predict3d_action_client;
  ros::Subscriber m_raw_data_subscriber;
  nbv_3d_cnn_msgs::FloatsConstPtr m_raw_data;
  ros::CallbackQueue m_raw_data_callback_queue;
};

#endif // AUTOCOMPLETE_PREDICT_H
