#include "projection_image_predict.h"

#include "evaluate_view.h"

#include <cv_bridge/cv_bridge.h>
#include <eigen_conversions/eigen_msg.h>

ProjectionImagePredict::ProjectionImagePredict(ros::NodeHandle & nh): m_nh(nh)
{
  std::string param_string;

  m_nh.param<std::string>(PARAM_NAME_PREDICT_PROJECTION_ACTION_NAME, param_string,
                          PARAM_DEFAULT_PREDICT_PROJECTION_ACTION_NAME);
  if (param_string.empty())
  {
    ROS_WARN("nbv_3d_cnn_evaluate_view: ImageProjectionPredict: empty prediction server name, "
             "predictions will not be possible.");
  }
  if (!param_string.empty())
  {
    m_projection_predict_action_client.reset(new ProjectionPredictActionClient(param_string, true));

    ROS_INFO("nbv_3d_cnn_evaluate_view: ImageProjectionPredict: waiting for prediction server...");
    m_projection_predict_action_client->waitForServer();
    ROS_INFO("nbv_3d_cnn_evaluate_view: ImageProjectionPredict: prediction server ok.");
  }

  m_nh.param<std::string>(PARAM_NAME_PIP_CROP_BBOX_MIN, param_string, PARAM_DEFAULT_PIP_CROP_BBOX_MIN);
  {
    std::istringstream istr(param_string);
    istr >> m_crop_bbox_min.x() >> m_crop_bbox_min.y() >> m_crop_bbox_min.z();
    if (!istr)
    {
      ROS_FATAL("nbv_3d_cnn_evaluate_view: ImageProjectionPredict: unable to parse bbox min: %s", param_string.c_str());
      exit(1);
    }
  }

  m_nh.param<std::string>(PARAM_NAME_PIP_CROP_BBOX_MAX, param_string, PARAM_DEFAULT_PIP_CROP_BBOX_MAX);
  {
    std::istringstream istr(param_string);
    istr >> m_crop_bbox_max.x() >> m_crop_bbox_max.y() >> m_crop_bbox_max.z();
    if (!istr)
    {
      ROS_FATAL("nbv_3d_cnn_evaluate_view: ImageProjectionPredict: unable to parse bbox max: %s", param_string.c_str());
      exit(1);
    }
  }
}

void ProjectionImagePredict::SetEnvironment(const Voxelgrid & environment,
                                            const Eigen::Vector3f & environment_origin,
                                            const float voxel_size)
{
  Eigen::Vector3i bbox_min_i =
      ((m_crop_bbox_min - environment_origin) / voxel_size).array().round().cast<int>();
  Eigen::Vector3i bbox_max_i =
      ((m_crop_bbox_max - environment_origin) / voxel_size).array().round().cast<int>();
  bbox_min_i = bbox_min_i.array().max(Eigen::Vector3i::Zero().array());
  bbox_max_i = bbox_max_i.array().min(environment.GetSize().array());
  ROS_INFO_STREAM("nbv_3d_cnn_evaluate_view: ImageProjectionPredict: cropped bounding box is " << bbox_min_i.transpose()
                  << " - " << bbox_max_i.transpose());
  const Voxelgrid cropped = *environment.GetSubmatrix(bbox_min_i, bbox_max_i - bbox_min_i + Eigen::Vector3i::Ones());

  m_environment_voxel_size = voxel_size * 2.0f;
  m_environment_origin = (bbox_min_i.cast<float>() + Eigen::Vector3f::Ones() * 0.5) * voxel_size + environment_origin;
  m_environment = cropped.HalveSize();
}

bool ProjectionImagePredict::Predict(const uint64 image_width,
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

                                     cv::Mat & probability_mask)
{
  nbv_3d_cnn_real_image_msgs::ProjectionPredictGoal goal;

  if (!m_projection_predict_action_client)
  {
    ROS_ERROR("nbv_3d_cnn_evaluate_view: ImageProjectionPredict: could not predict, server not initialized at startup.");
    return false;
  }

  if (!m_environment)
  {
    ROS_ERROR("nbv_3d_cnn_evaluate_view: ImageProjectionPredict: environment not set!");
    return false;
  }

  const Voxelgrid & environment = *m_environment;
  const float environment_voxel_size = m_environment_voxel_size;
  const Eigen::Vector3f & environment_origin = m_environment_origin;

  tf::poseEigenToMsg(pose.cast<double>(), goal.camera_pose);

  goal.environment_width = environment.GetWidth();
  goal.environment_height = environment.GetHeight();
  goal.environment_depth = environment.GetDepth();
  goal.environment_voxel_size = environment_voxel_size;

  goal.with_saliency_images = with_saliency_images;

  tf::pointEigenToMsg(environment_origin.cast<double>(), goal.environment_origin);

  goal.ternary_voxelgrid = environment.ToFloatVector();

  goal.image_width = image_width;
  goal.image_height = image_height;
  goal.image_depth = image_depth;
  goal.max_range = max_range;
  goal.fx = fx;
  goal.fy = fy;
  goal.cx = cx;
  goal.cy = cy;

  goal.output_mask.resize(goal.image_width * goal.image_height);
  for (uint64 y = 0; y < image_height; y++)
    for (uint64 x = 0; x < image_width; x++)
      goal.output_mask[x + y * image_width] = output_mask.at<float>(y, x);

  cv_bridge::CvImagePtr cv_bridge(new cv_bridge::CvImage);

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

  ROS_INFO("nbv_3d_cnn_evaluate_view: ImageProjectionPredict: sending goal...");

  m_projection_predict_action_client->sendGoal(goal);

  ROS_INFO("nbv_3d_cnn_evaluate_view: ImageProjectionPredict: waiting for result...");
  m_projection_predict_action_client->waitForResult();

  const actionlib::SimpleClientGoalState state = m_projection_predict_action_client->getState();
  ROS_INFO("nbv_3d_cnn_evaluate_view: action state is %s", state.toString().c_str());
  if (state != actionlib::SimpleClientGoalState::SUCCEEDED)
  {
    ROS_ERROR("nbv_3d_cnn_evaluate_view: action did not succeed!");
    return false;
  }

  nbv_3d_cnn_real_image_msgs::ProjectionPredictResult result = *m_projection_predict_action_client->getResult();

  ROS_INFO("nbv_3d_cnn_evaluate_view: ImageProjectionPredict: got image with shape %u %u", unsigned(result.image_height),
           unsigned(result.image_width));

  cv_bridge = cv_bridge::toCvCopy(result.probability_mask);
  probability_mask = cv_bridge->image;

  if (!probability_mask.data)
  {
    ROS_INFO("nbv_3d_cnn_evaluate_view: ImageProjectionPredict: returned image is empty!");
    return false;
  }

  m_last_saliency_images.clear();
  if (with_saliency_images)
    m_last_saliency_images = result.saliency_images;

  m_last_prediction_time = result.prediction_time;

  return true;
}
