#include "evaluate_view.h"

#include "evaluate_view_opencl.h"

#include "evaluate_view_filter.h"

#include "autocomplete_predict.h"
#include "image_predict.h"
#include "projection_image_predict.h"

#include <rmonica_voxelgrid_common/voxelgrid.h>
#include <render_robot_urdf/RenderRobotUrdf.h>

// ROS
#include <ros/ros.h>
#include <actionlib/server/simple_action_server.h>
#include <nbv_3d_cnn_evaluate_view_msgs/SetEnvironmentAction.h>
#include <nbv_3d_cnn_evaluate_view_msgs/EvaluateViewAction.h>
#include <nbv_3d_cnn_evaluate_view_msgs/RawProjectionAction.h>
#include <nbv_3d_cnn_evaluate_view_msgs/GroundTruthEvaluateViewAction.h>
#include <nbv_3d_cnn_evaluate_view_msgs/GenerateFilterCircleRectangleMaskAction.h>
#include <eigen_conversions/eigen_msg.h>
#include <pcl_conversions/pcl_conversions.h>
#include <cv_bridge/cv_bridge.h>

#include <Eigen/Dense>
#include <Eigen/StdVector>

// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

// PCL
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

#include <memory>
#include <vector>
#include <string>
#include <mutex>
#include <stdint.h>

class NBV3DCNNEvaluateView
{
  public:
  typedef uint64_t uint64;
  typedef uint8_t uint8;
  typedef int8_t int8;

  typedef rmonica_voxelgrid_common::Voxelgrid Voxelgrid;

  typedef actionlib::SimpleActionServer<nbv_3d_cnn_evaluate_view_msgs::SetEnvironmentAction> EnvironmentActionServer;
  typedef std::shared_ptr<EnvironmentActionServer> EnvironmentActionServerPtr;

  typedef actionlib::SimpleActionServer<nbv_3d_cnn_evaluate_view_msgs::RawProjectionAction> RawProjectionActionServer;
  typedef std::shared_ptr<RawProjectionActionServer> RawProjectionActionServerPtr;

  typedef actionlib::SimpleActionServer<nbv_3d_cnn_evaluate_view_msgs::EvaluateViewAction> ViewActionServer;
  typedef std::shared_ptr<ViewActionServer> ViewActionServerPtr;

  typedef actionlib::SimpleActionServer<nbv_3d_cnn_evaluate_view_msgs::GroundTruthEvaluateViewAction> GTViewActionServer;
  typedef std::shared_ptr<GTViewActionServer> GTViewActionServerPtr;

  typedef actionlib::SimpleActionServer<nbv_3d_cnn_evaluate_view_msgs::GenerateFilterCircleRectangleMaskAction>
    GenerateFilterCircleRectangleMaskServer;
  typedef std::shared_ptr<GenerateFilterCircleRectangleMaskServer> GenerateFilterCircleRectangleMaskServerPtr;

  typedef pcl::PointCloud<pcl::PointXYZI> PointXYZICloud;
  typedef std::vector<Eigen::Vector4i, Eigen::aligned_allocator<Eigen::Vector4i> > Vector4iVector;

  typedef std::vector<float> FloatVector;
  typedef std::vector<int8> Int8Vector;

  NBV3DCNNEvaluateView(ros::NodeHandle & nh):
    m_nh(nh), m_opencl(m_nh), m_view_filter(m_nh), m_autocomplete_predict(m_nh), m_image_predict(m_nh),
    m_projection_image_predict(m_nh)
  {
    std::string param_string;
    double param_double;
    int param_int;

    m_nh.param<double>(PARAM_NAME_AP_LOST_IF_OCCUPIED, param_double, PARAM_DEFAULT_AP_LOST_IF_OCCUPIED);
    m_a_priori_ray_lost_if_occupied = param_double;

    m_nh.param<double>(PARAM_NAME_AP_OCCUPIED_IF_OUTSIDE, param_double, PARAM_DEFAULT_AP_OCCUPIED_IF_OUTSIDE);
    m_a_priori_ray_occupied_if_outside = param_double;

    m_nh.param<double>(PARAM_NAME_AP_OCCUPIED, param_double, PARAM_DEFAULT_AP_OCCUPIED);
    m_a_priori_occupancy_probability = param_double;

    m_nh.param<double>(PARAM_NAME_ENVIRONMENT_ORIGIN_OFFSET, param_double, PARAM_DEFAULT_ENVIRONMENT_ORIGIN_OFFSET);
    m_environment_origin_offset = param_double;

    m_nh.param<double>(PARAM_NAME_CNN_MAX_RANGE, param_double, PARAM_DEFAULT_CNN_MAX_RANGE);
    m_cnn_max_range = param_double;

    m_nh.param<int>(PARAM_NAME_CNN_IMAGE_DEPTH, param_int, PARAM_DEFAULT_CNN_IMAGE_DEPTH);
    m_cnn_image_depth = param_int;

    m_nh.param<std::string>(PARAM_NAME_SET_ENVIRONMENT_ACTION_NAME, param_string, PARAM_DEFAULT_SET_ENVIRONMENT_ACTION_NAME);
    m_environment_action_server.reset(new EnvironmentActionServer(m_nh, param_string,
                                                                  boost::bind(&NBV3DCNNEvaluateView::onEnvironmentAction,
                                                                            this, _1),
                                                                  false));

    m_nh.param<std::string>(PARAM_NAME_EVALUATE_VIEW_ACTION_NAME, param_string, PARAM_DEFAULT_EVALUATE_VIEW_ACTION_NAME);
    m_evaluate_view_action_server.reset(new ViewActionServer(m_nh, param_string,
                                                             boost::bind(&NBV3DCNNEvaluateView::onEvaluateViewAction,
                                                                       this, _1),
                                                             false));

    m_nh.param<std::string>(PARAM_NAME_RAW_PROJECTION_ACTION_NAME, param_string, PARAM_DEFAULT_RAW_PROJECTION_ACTION_NAME);
    m_raw_projection_action_server.reset(new RawProjectionActionServer(m_nh, param_string,
                                               boost::bind(&NBV3DCNNEvaluateView::onRawProjectionAction,
                                                         this, _1),
                                               false));

    m_nh.param<std::string>(PARAM_NAME_GROUND_TRUTH_EV_ACTION_NAME, param_string, PARAM_DEFAULT_GROUND_TRUTH_EV_ACTION_NAME);
    m_ground_truth_evaluate_view_action_server.reset(new GTViewActionServer(m_nh, param_string,
                                                     boost::bind(&NBV3DCNNEvaluateView::onGroundTruthEvaluateViewAction,
                                                               this, _1),
                                                     false));


    m_nh.param<std::string>(PARAM_NAME_CIRCLE_FILTER_ACTION_NAME, param_string, PARAM_DEFAULT_CIRCLE_FILTER_ACTION_NAME);
    m_generate_filter_circle_rectangle_mask_server.reset(new GenerateFilterCircleRectangleMaskServer(
                                                           m_nh, param_string,
                                                           boost::bind(&NBV3DCNNEvaluateView::onCircleFilterAction,
                                                                        this, _1),
                                                                        false));

    m_nh.param<std::string>(PARAM_NAME_RENDER_ROBOT_URDF_SERVICE, param_string, PARAM_DEFAULT_RENDER_ROBOT_URDF_SERVICE);
    m_render_robot_service_client = nh.serviceClient<render_robot_urdf::RenderRobotUrdf>(param_string);

    {
      std::lock_guard<std::mutex> lock(m_mutex);
      m_environment_action_server->start();
      m_evaluate_view_action_server->start();
      m_raw_projection_action_server->start();
      m_generate_filter_circle_rectangle_mask_server->start();
      m_ground_truth_evaluate_view_action_server->start();
    }
  }

  void onEnvironmentAction(const nbv_3d_cnn_evaluate_view_msgs::SetEnvironmentGoalConstPtr & goal)
  {
    ROS_INFO("nbv_3d_cnn_evaluate_view: onEnvironmentAction: start.");
    std::lock_guard<std::mutex> lock(m_mutex);

    const Eigen::Vector3i cnn_matrix_size(goal->probabilities_size[0],
                                          goal->probabilities_size[1],
                                          goal->probabilities_size[2]);
    const Eigen::Vector3i cnn_matrix_origin(goal->probabilities_origin[0],
                                            goal->probabilities_origin[1],
                                            goal->probabilities_origin[2]);
    const float cnn_matrix_voxel_size = goal->probabilities_voxel_size;
    const float voxel_size = goal->voxel_size;
    const float absolute_origin_offset = voxel_size * m_environment_origin_offset;
    const Eigen::Vector3f environment_origin(goal->environment_origin.x,
                                             goal->environment_origin.y,
                                             goal->environment_origin.z);
    const Eigen::Vector3f offset_environment_origin(goal->environment_origin.x + absolute_origin_offset,
                                                    goal->environment_origin.y + absolute_origin_offset,
                                                    goal->environment_origin.z + absolute_origin_offset);

    Voxelgrid::Ptr vg = Voxelgrid::FromInt8MultiArray(goal->environment);
    m_current_environment = vg;
    m_voxel_size = voxel_size;

    const Eigen::Vector3i environment_size = vg->GetSize();
    Int8Vector environment(environment_size.prod());
    for (uint64 z = 0; z < environment_size.z(); z++)
      for (uint64 y = 0; y < environment_size.y(); y++)
        for (uint64 x = 0; x < environment_size.x(); x++)
        {
          const uint64 i3 = x + y * environment_size.x() + z * environment_size.x() * environment_size.y();
          environment[i3] = vg->at(x, y, z);
        }

    const ros::Time prediction_start_time = ros::Time::now();
    Voxelgrid::Ptr prob_vg = Voxelgrid::Ptr(new Voxelgrid(cnn_matrix_size));
    if (cnn_matrix_size != Eigen::Vector3i::Zero())
    {
      Voxelgrid::Ptr autocompleted(new Voxelgrid);
      Voxelgrid empty;
      Voxelgrid occupied;
      Voxelgrid downsampled_env;

      const Eigen::Vector3i downsampled_cnn_matrix_size = (cnn_matrix_size.cast<float>() * voxel_size /
                                                           cnn_matrix_voxel_size + 0.5f * Eigen::Vector3f::Ones()).cast<int>();

      m_autocomplete_predict.DownsampleKinfuVoxelgrid(*vg,
                                                      voxel_size,
                                                      cnn_matrix_voxel_size,
                                                      cnn_matrix_origin,
                                                      downsampled_cnn_matrix_size,
                                                      empty,
                                                      occupied,
                                                      downsampled_env
                                                      );
      m_autocomplete_predict.Predict3d(empty, occupied, *autocompleted);

      for (uint64 z = 0; z < cnn_matrix_size.z(); z++)
        for (uint64 y = 0; y < cnn_matrix_size.y(); y++)
          for (uint64 x = 0; x < cnn_matrix_size.x(); x++)
          {
            const Eigen::Vector3i index(x, y, z);

            if (vg->at(index) > 0.5f)
            {
              prob_vg->at(index) = 1.0f;
              continue;
            }
            if (vg->at(index) < -0.5f)
            {
              prob_vg->at(index) = 0.0f;
              continue;
            }

            const Eigen::Vector3i autocompl_index = (index.cast<float>() * voxel_size / cnn_matrix_voxel_size).cast<int>();

            if ((autocompl_index.array() >= downsampled_cnn_matrix_size.array()).any())
            {
              prob_vg->at(index) = 0.0f;
              continue;
            }

            if (vg->at(index) == 0.0f)
              prob_vg->at(index) = autocompleted->at(autocompl_index);
          }
    }
    const ros::Duration prediction_time = ros::Time::now() - prediction_start_time;

    FloatVector probabilities(cnn_matrix_size.prod());
    for (uint64 z = 0; z < cnn_matrix_size.z(); z++)
      for (uint64 y = 0; y < cnn_matrix_size.y(); y++)
        for (uint64 x = 0; x < cnn_matrix_size.x(); x++)
        {
          const uint64 i3 = x + y * cnn_matrix_size.x() + z * cnn_matrix_size.x() * cnn_matrix_size.y();
          probabilities[i3] = prob_vg->at(x, y, z);
        }

    m_environment_origin = offset_environment_origin;

    m_current_probabilities = prob_vg;

    PointXYZICloud tsdf_cloud;
    pcl::fromROSMsg(goal->tsdf_cloud, tsdf_cloud);
    m_current_tsdf.resize(tsdf_cloud.size());
    for (uint64 i = 0; i < tsdf_cloud.size(); i++)
    {
      const Eigen::Vector3f vf(tsdf_cloud[i].x, tsdf_cloud[i].y, tsdf_cloud[i].z);
      const float intensity = tsdf_cloud[i].intensity;
      const Eigen::Vector3i vi = ((vf - environment_origin) / goal->voxel_size).array().round().cast<int>();
      const Eigen::Vector4i vv(vi.x(), vi.y(), vi.z(), intensity * 256);
      m_current_tsdf[i] = vv;
    }
    ROS_INFO("nbv_3d_cnn_evaluate_view:   tsdf has %u points.", unsigned(m_current_tsdf.size()));

    m_opencl.SetEnvironment(environment_size, environment,
                            m_a_priori_occupancy_probability,
                            cnn_matrix_origin, cnn_matrix_size, probabilities,
                            m_current_tsdf);

    m_projection_image_predict.SetEnvironment(*vg, environment_origin, voxel_size);

    nbv_3d_cnn_evaluate_view_msgs::SetEnvironmentResult result;
    result.predicted_probabilities = m_current_probabilities->ToFloat32MultiArray();
    result.ok = true;
    result.prediction_time = prediction_time.toSec();
    m_environment_action_server->setSucceeded(result);
    ROS_INFO("nbv_3d_cnn_evaluate_view: onEnvironmentAction: end.");
  }

  EvaluateViewOpenCL::MaskPixelVector EVFReachingResultToEVOMask(const EvaluateViewFilter::ReachingResult & reaching_result,
                                                                 const float voxel_size)
  {
    const uint64 width = reaching_result.depth.cols;
    const uint64 height = reaching_result.depth.rows;

    EvaluateViewOpenCL::MaskPixelVector mask(width * height);
    for (uint64 y = 0; y < height; y++)
      for (uint64 x = 0; x < width; x++)
      {
        EvaluateViewOpenCL::MaskPixel & cl_pix = mask[x + y * width];
        const float depth = reaching_result.depth.at<float>(y, x);
        const uint8 status = reaching_result.status.at<uint8>(y, x);

        cl_pix.depth = depth / voxel_size;
        uint8 s;
        switch (status)
        {
          case EvaluateViewFilter::ReachingResult::LOST:     s = EvaluateViewOpenCL::MaskPixel::STATUS_LOST;     break;
          case EvaluateViewFilter::ReachingResult::UNKNOWN:  s = EvaluateViewOpenCL::MaskPixel::STATUS_UNKNOWN;  break;
          case EvaluateViewFilter::ReachingResult::OCCUPIED: s = EvaluateViewOpenCL::MaskPixel::STATUS_OCCUPIED; break;
          default:
            ROS_FATAL("evaluate_view: EVFReachingResultToEVOMask: unknown status %u", unsigned(status));
            exit(1);
        }
        cl_pix.visibility = s;
      }

    return mask;
  }

  void onCircleFilterAction(const nbv_3d_cnn_evaluate_view_msgs::GenerateFilterCircleRectangleMaskGoalConstPtr & goal)
  {
    ROS_INFO("nbv_3d_cnn_evaluate_view: onCircleFilterAction: start.");
    const uint64 width = goal->width;
    const uint64 height = goal->height;

    const cv::Mat cf = m_view_filter.GenerateFilterCircleRectangleMask(width, height);

    cv_bridge::CvImagePtr cv_bridge(new cv_bridge::CvImage);
    cv_bridge->image = cf;
    cv_bridge->encoding = "8UC1";

    nbv_3d_cnn_evaluate_view_msgs::GenerateFilterCircleRectangleMaskResult result;
    result.mask = *cv_bridge->toImageMsg();

    m_generate_filter_circle_rectangle_mask_server->setSucceeded(result);
    ROS_INFO("nbv_3d_cnn_evaluate_view: onCircleFilterAction: end.");
  }

  void onGroundTruthEvaluateViewAction(const nbv_3d_cnn_evaluate_view_msgs::GroundTruthEvaluateViewGoalConstPtr & goal)
  {
    ROS_INFO("nbv_3d_cnn_evaluate_view: onGroundTruthEvaluateViewAction: start.");

    if (!m_current_environment)
    {
      ROS_ERROR("nbv_3d_cnn_evaluate_view: onGroundTruthEvaluateViewAction: current environment not set.");
      m_ground_truth_evaluate_view_action_server->setAborted();
      return;
    }

    Eigen::Affine3d pose_d;
    tf::poseMsgToEigen(goal->pose, pose_d);
    pose_d.translation() -= m_environment_origin.cast<double>();
    pose_d.translation() /= m_voxel_size;

    const uint64 width = goal->size_x;
    const uint64 height = goal->size_y;

    const Eigen::Affine3f pose = pose_d.cast<float>();

    const Eigen::Vector2f center(goal->center_x, goal->center_y);
    const Eigen::Vector2f focal(goal->focal_x, goal->focal_y);
    const Eigen::Vector2i image_size(width, height);

    cv_bridge::CvImagePtr cv_bridge;
    cv_bridge = cv_bridge::toCvCopy(goal->depth_image);
    cv::Mat actual_sensor_image = cv_bridge->image;
    if (actual_sensor_image.rows != height || actual_sensor_image.cols != width)
    {
      ROS_ERROR("nbv_3d_cnn_evaluate_view: onGroundTruthEvaluateViewAction: expected image size %u %u, got %u %u.",
                unsigned(width), unsigned(height),
                unsigned(actual_sensor_image.cols), unsigned(actual_sensor_image.rows));
      m_ground_truth_evaluate_view_action_server->setAborted();
      return;
    }

    FloatVector opencl_sensor_image(actual_sensor_image.rows * actual_sensor_image.cols);
    for (uint64 y = 0; y < height; y++)
      for (uint64 x = 0; x < width; x++)
      {
        const uint64 i = x + y * width;
        opencl_sensor_image[i] = actual_sensor_image.at<float>(y, x) / m_voxel_size;
      }

    EvaluateViewOpenCL::ROIData roi_data;
    roi_data.has_sphere_roi = goal->has_roi_sphere;
    roi_data.sphere_center = Eigen::Vector3f(goal->roi_sphere_center.x,
                                             goal->roi_sphere_center.y,
                                             goal->roi_sphere_center.z);
    roi_data.sphere_center -= m_environment_origin;
    roi_data.sphere_center /= m_voxel_size;
    roi_data.sphere_radius = goal->roi_sphere_radius;
    roi_data.sphere_radius /= m_voxel_size;

    FloatVector scores;
    scores = m_opencl.ComputeGroundTruthScores(center, focal, image_size, opencl_sensor_image,
                                               roi_data, pose);

    nbv_3d_cnn_evaluate_view_msgs::GroundTruthEvaluateViewResult result;

    float gain = 0.0;
    cv::Mat scores_img(height, width, CV_32FC1);
    for (uint64 y = 0; y < height; y++)
      for (uint64 x = 0; x < width; x++)
      {
        const uint64 i = x + y * width;
        scores_img.at<float>(y, x) = scores[i];
        gain += scores[i];
      }
    result.gain = gain;

    cv_bridge->image = scores_img;
    cv_bridge->encoding = "32FC1";
    result.predicted_score_image = *cv_bridge->toImageMsg();

    m_ground_truth_evaluate_view_action_server->setSucceeded(result);
    ROS_INFO("nbv_3d_cnn_evaluate_view: onGroundTruthEvaluateViewAction: end.");
  }

  void onRawProjectionAction(const nbv_3d_cnn_evaluate_view_msgs::RawProjectionGoalConstPtr & goal)
  {
    ROS_INFO("nbv_3d_cnn_evaluate_view: onRawProjectionAction: start.");
    std::lock_guard<std::mutex> lock(m_mutex);

    if (!m_current_environment)
    {
      ROS_ERROR("nbv_3d_cnn_evaluate_view: onRawProjectionAction: current environment not set.");
      m_raw_projection_action_server->setAborted();
      return;
    }

    Eigen::Affine3d pose_d;
    tf::poseMsgToEigen(goal->pose, pose_d);
    pose_d.translation() -= m_environment_origin.cast<double>();
    pose_d.translation() /= m_voxel_size;

    const uint64 width = goal->size_x;
    const uint64 height = goal->size_y;

    const Eigen::Affine3f pose = pose_d.cast<float>();
    const Eigen::Vector2i size(goal->size_x, goal->size_y);
    const Eigen::Vector2f center(goal->center_x, goal->center_y);
    const Eigen::Vector2f focal(goal->focal_x, goal->focal_y);

    const sensor_msgs::JointState joint_state(goal->joint_state);

    const float min_range = goal->min_range;
    const float max_range_cells = goal->max_range / m_voxel_size;
    const float min_range_cells = goal->min_range / m_voxel_size;

    const bool predict_robot_image = goal->predict_robot_image;

    ROS_INFO("nbv_3d_cnn_evaluate_view: onRawProjectionAction: ray casting...");
    EvaluateViewOpenCL::RaycastResult raycast_result;
    raycast_result = m_opencl.Raycast(center,
                                      focal, size, max_range_cells, pose);

    ROS_INFO("nbv_3d_cnn_evaluate_view: onRawProjectionAction: getting robot depth image...");
    cv::Mat robot_depth_image, robot_normal_image;
    if (predict_robot_image)
    {
      Eigen::Affine3d robot_pose_d;
      tf::poseMsgToEigen(goal->pose, robot_pose_d);
      GetRobotDepthImage(robot_pose_d, joint_state, center, focal, size,
                         goal->max_range, robot_depth_image, robot_normal_image);
    }

    ROS_INFO("nbv_3d_cnn_evaluate_view: onRawProjectionAction: sending result...");
    nbv_3d_cnn_evaluate_view_msgs::RawProjectionResult result;

    cv_bridge::CvImagePtr cv_bridge(new cv_bridge::CvImage);
    {
      cv::Mat depth_image = cv::Mat(height, width, CV_32FC1);
      cv::Mat normal_image = cv::Mat(height, width, CV_32FC3);
      RaycastResultToImages(raycast_result, depth_image, normal_image);

      cv_bridge->image = depth_image;
      cv_bridge->encoding = "32FC1";
      result.predicted_depth_image = *cv_bridge->toImageMsg();
      cv_bridge->image = normal_image;
      cv_bridge->encoding = "32FC3";
      result.predicted_normal_image = *cv_bridge->toImageMsg();
    }

    if (predict_robot_image)
    {
      cv_bridge->image = robot_depth_image;
      cv_bridge->encoding = "32FC1";
      result.predicted_robot_image = *cv_bridge->toImageMsg();
      cv_bridge->image = robot_normal_image;
      cv_bridge->encoding = "32FC3";
      result.predicted_robot_normal_image = *cv_bridge->toImageMsg();
    }

    m_raw_projection_action_server->setSucceeded(result);
    ROS_INFO("nbv_3d_cnn_evaluate_view: onRawProjectionAction: end.");
  }

  void RaycastResultToImages(const EvaluateViewOpenCL::RaycastResult & raycast_result,
                             cv::Mat & depth_image, cv::Mat & normal_image)
  {
    const uint64 width = raycast_result.size.x();
    const uint64 height = raycast_result.size.y();
    depth_image = cv::Mat(height, width, CV_32FC1);
    normal_image = cv::Mat(height, width, CV_32FC3);
    for (uint64 y = 0; y < height; y++)
      for (uint64 x = 0; x < width; x++)
      {
        const uint64 i2 = x + y * width;
        const EvaluateViewOpenCL::CellResult & ray = raycast_result.ray_results[i2];

        if (ray.status != EvaluateViewOpenCL::CellResult::OCCUPIED)
        {
          depth_image.at<float>(y, x) = 0.0f;
          normal_image.at<cv::Vec3f>(y, x) = cv::Vec3f(0.0f, 0.0f, 0.0f);
        }
        else
        {
          depth_image.at<float>(y, x) = ray.z * m_voxel_size * ray.local_direction.z();
          normal_image.at<cv::Vec3f>(y, x) = cv::Vec3f(ray.normal.x(), ray.normal.y(), ray.normal.z());
        }
      }
  }

  void onEvaluateViewAction(const nbv_3d_cnn_evaluate_view_msgs::EvaluateViewGoalConstPtr & goal)
  {
    ROS_INFO("nbv_3d_cnn_evaluate_view: onEvaluateViewAction: start.");
    std::lock_guard<std::mutex> lock(m_mutex);

    if (!m_current_environment)
    {
      ROS_ERROR("nbv_3d_cnn_evaluate_view: onEvaluateViewAction: current environment not set.");
      m_evaluate_view_action_server->setAborted();
      return;
    }

    Eigen::Affine3d pose_d;
    tf::poseMsgToEigen(goal->pose, pose_d);
    pose_d.translation() -= m_environment_origin.cast<double>();
    pose_d.translation() /= m_voxel_size;

    const uint64 width = goal->size_x;
    const uint64 height = goal->size_y;

    const float a_priori_ray_lost_if_occupied = m_a_priori_ray_lost_if_occupied;
    const float a_priori_ray_occupied_if_outside = m_a_priori_ray_occupied_if_outside;

    const Eigen::Affine3f pose = pose_d.cast<float>();
    const Eigen::Vector2i size(goal->size_x, goal->size_y);
    const Eigen::Vector2f center(goal->center_x, goal->center_y);
    const Eigen::Vector2f focal(goal->focal_x, goal->focal_y);
    const sensor_msgs::JointState joint_state(goal->joint_state);

    const bool with_saliency_images = goal->with_saliency_images;

    const float min_range = goal->min_range;
    const float max_range_cells = goal->max_range / m_voxel_size;
    const float min_range_cells = goal->min_range / m_voxel_size;

    const float fixed_probability = goal->fixed_probability;

    EvaluateViewOpenCL::ROIData roi_data;
    roi_data.has_sphere_roi = goal->has_roi_sphere;
    roi_data.sphere_center = Eigen::Vector3f(goal->roi_sphere_center.x,
                                             goal->roi_sphere_center.y,
                                             goal->roi_sphere_center.z);
    roi_data.sphere_center -= m_environment_origin;
    roi_data.sphere_center /= m_voxel_size;
    roi_data.sphere_radius = goal->roi_sphere_radius;
    roi_data.sphere_radius /= m_voxel_size;

    uint64 mode = goal->mode;

    cv::Mat external_mask;
    if (goal->external_mask.data.size())
    {
      cv_bridge::CvImagePtr cv_bridge;
      cv_bridge = cv_bridge::toCvCopy(goal->external_mask);
      external_mask = cv_bridge->image;
      if (!external_mask.data)
      {
        ROS_ERROR("nbv_3d_cnn_evaluate_view: onEvaluateViewAction: could not decode external mask.");
        m_evaluate_view_action_server->setAborted();
        return;
      }
    }

    // ---- PROCESSING ----

    double network_prediction_time = 0.0;

    const ros::Time processing_time = ros::Time::now();

    const ros::Time raycast_time = ros::Time::now();

    EvaluateViewOpenCL::RaycastResult rays;
    if (mode == goal->MODE_ADVANCED_PROB ||
        mode == goal->MODE_IMAGE_PREDICT_PROB ||
        mode == goal->MODE_PROJ_PREDICT_PROB)
    {
      rays = m_opencl.Raycast(center,
                              focal, size, max_range_cells, pose);
    }

    const ros::Duration raycast_duration = ros::Time::now() - raycast_time;

    const ros::Time robot_time = ros::Time::now();

    cv::Mat robot_depth_image = cv::Mat(size.y(), size.x(), CV_32FC1, 0.0);
    cv::Mat robot_normal_image = cv::Mat(size.y(), size.x(), CV_32FC3, cv::Vec3f(0.0, 0.0, 0.0));
    if (mode == goal->MODE_ADVANCED_PROB ||
        mode == goal->MODE_IMAGE_PREDICT_PROB ||
        mode == goal->MODE_PROJ_PREDICT_PROB)
    {
      if (joint_state.position.size())
      {
        Eigen::Affine3d robot_pose_d;
        tf::poseMsgToEigen(goal->pose, robot_pose_d);
        GetRobotDepthImage(robot_pose_d, joint_state, center, focal, size,
                           goal->max_range, robot_depth_image, robot_normal_image);
      }
    }

    const ros::Duration robot_duration = ros::Time::now() - robot_time;

    const ros::Time filter_time = ros::Time::now();

    const cv::Mat circle_rectangle_mask = m_view_filter.GenerateFilterCircleRectangleMask(width, height);

    EvaluateViewFilter::ReachingResultPtr maybe_reaching_result;
    if (mode == goal->MODE_ADVANCED_PROB)
    {
      const EvaluateViewFilter::ReachingResult reaching_result = m_view_filter.Filter(
                                                                                rays, m_voxel_size, center, focal, min_range,
                                                                                robot_depth_image, robot_normal_image);
      maybe_reaching_result.reset(new EvaluateViewFilter::ReachingResult(reaching_result));
    }
    if (mode == goal->MODE_IMAGE_PREDICT_PROB)
    {
      cv::Mat depth_image = cv::Mat(height, width, CV_32FC1);
      cv::Mat normal_image = cv::Mat(height, width, CV_32FC3);
      RaycastResultToImages(rays, depth_image, normal_image);

      cv::Mat circle_rectangle_mask_float;
      circle_rectangle_mask.convertTo(circle_rectangle_mask_float, CV_32FC1);

      if (!m_image_predict.Predict(depth_image, normal_image, robot_depth_image, robot_normal_image,
                                   circle_rectangle_mask_float, external_mask))
      {
        ROS_FATAL("nbv_3d_cnn_evaluate_view: image predict failed!");
        exit(1);
      }

      network_prediction_time = m_image_predict.GetLastPredictionTime();
    }
    if (mode == goal->MODE_PROJ_PREDICT_PROB)
    {
      cv::Mat depth_image = cv::Mat(height, width, CV_32FC1);
      cv::Mat normal_image = cv::Mat(height, width, CV_32FC3);
      RaycastResultToImages(rays, depth_image, normal_image);

      cv::Mat circle_rectangle_mask_float;
      circle_rectangle_mask.convertTo(circle_rectangle_mask_float, CV_32FC1);

      Eigen::Affine3d cam_pose_d;
      tf::poseMsgToEigen(goal->pose, cam_pose_d);

      const float max_range = m_cnn_max_range;
      const uint64 image_depth = m_cnn_image_depth;

      if (!m_projection_image_predict.Predict(width, height, image_depth,
                                              focal.x(), focal.y(), center.x(), center.y(),
                                              max_range,
                                              with_saliency_images,
                                              cam_pose_d.cast<float>(),
                                              depth_image, normal_image, robot_depth_image, robot_normal_image,
                                              circle_rectangle_mask_float, external_mask))
      {
        ROS_FATAL("nbv_3d_cnn_evaluate_view: projection image predict failed!");
        exit(1);
      }

      network_prediction_time = m_projection_image_predict.GetLastPredictionTime();
    }

    const ros::Duration filter_duration = ros::Time::now() - filter_time;

    const ros::Time raycast2_time = ros::Time::now();

    FloatVector scores;
    FloatVector unmasked_scores;
    if (mode == goal->MODE_ADVANCED_PROB)
    {
      const EvaluateViewOpenCL::MaskPixelVector mask = EVFReachingResultToEVOMask(*maybe_reaching_result, m_voxel_size);

      unmasked_scores = m_opencl.RaycastProbabilistic(center,
                                                      focal,
                                                      size,
                                                      mask,
                                                      max_range_cells,
                                                      min_range_cells,
                                                      a_priori_ray_lost_if_occupied,
                                                      a_priori_ray_occupied_if_outside,
                                                      roi_data,
                                                      pose
                                                      );
    }
    if (mode == goal->MODE_STANDARD_PROB ||
        mode == goal->MODE_FIXED ||
        mode == goal->MODE_IMAGE_PREDICT_PROB ||
        mode == goal->MODE_PROJ_PREDICT_PROB)
    {
      unmasked_scores = m_opencl.RaycastDumbProbabilistic(center,
                                                          focal,
                                                          size,
                                                          max_range_cells,
                                                          min_range_cells,
                                                          a_priori_ray_occupied_if_outside,
                                                          roi_data,
                                                          pose
                                                          );

      for (uint64 y = 0; y < height; y++)
        for (uint64 x = 0; x < width; x++)
          if (!circle_rectangle_mask.at<uint8>(y, x))
            unmasked_scores[x + y * width] = 0.0f;
      scores = unmasked_scores;
      if (external_mask.data)
      {
        for (uint64 y = 0; y < height; y++)
          for (uint64 x = 0; x < width; x++)
            scores[x + y * width] *= external_mask.at<float>(y, x);
      }

      if (mode == goal->MODE_FIXED)
      {
        for (uint64 y = 0; y < height; y++)
          for (uint64 x = 0; x < width; x++)
            scores[x + y * width] *= fixed_probability;
      }
    }

    const ros::Duration raycast2_duration = ros::Time::now() - raycast2_time;

    const ros::Duration processing_duration = ros::Time::now() - processing_time;

    ROS_INFO_STREAM("nbv_3d_cnn_evaluate_view: time:\n"
                    << "    Raycast : " << raycast_duration.toSec()  << " s\n"
                    << "    Robot   : " << robot_duration.toSec()    << " s\n"
                    << "    Filter  : " << filter_duration.toSec()   << " s\n"
                    << "    Raycast2: " << raycast2_duration.toSec() << " s\n"
                    << "    TOTAL   : " << processing_duration.toSec() << " s");

    ROS_INFO_STREAM("nbv_3d_cnn_evaluate_view: of which network prediction time was " << network_prediction_time << " s");

    // ---- SENDING RESULT ----

    nbv_3d_cnn_evaluate_view_msgs::EvaluateViewResult result;

    result.raycast_time = raycast_duration.toSec();
    result.robot_time = robot_duration.toSec();
    result.filter_time = filter_duration.toSec();
    result.prob_raycast_time = raycast2_duration.toSec();
    result.total_time = processing_duration.toSec();

    result.network_prediction_time = network_prediction_time;

    if (with_saliency_images && mode == goal->MODE_PROJ_PREDICT_PROB)
    {
      result.saliency_images = m_projection_image_predict.GetLastSaliencyImages();
    }

    float total_score = 0.0;
    for (const float s : scores)
      total_score += s;
    result.gain = total_score;

    cv_bridge::CvImagePtr cv_bridge(new cv_bridge::CvImage);
    {
      cv::Mat predicted_depth;
      if (maybe_reaching_result)
      {
        const EvaluateViewFilter::ReachingResult & reaching_result = *maybe_reaching_result;
        predicted_depth = reaching_result.depth.clone();
        for (uint64 y = 0; y < size.y(); y++)
          for (uint64 x = 0; x < size.x(); x++)
          {
            if (reaching_result.status.at<uint8>(y, x) != EvaluateViewFilter::ReachingResult::OCCUPIED)
              predicted_depth.at<float>(y, x) = 0.0;
          }
      }
      else
      {
        predicted_depth = cv::Mat(size.y(), size.x(), CV_32FC1, 0.0f);
      }
      cv_bridge->image = predicted_depth;
      cv_bridge->encoding = "32FC1";
      result.predicted_realistic_image = *cv_bridge->toImageMsg();
    }

    {
      cv::Mat predicted_mask = cv::Mat(height, width, CV_32FC1);
      const float fixed_prob = (mode == goal->MODE_FIXED) ? fixed_probability : 1.0f;
      if (maybe_reaching_result)
      {
        const EvaluateViewFilter::ReachingResult & reaching_result = *maybe_reaching_result;
        predicted_mask = 0.0f;
        for (uint64 y = 0; y < size.y(); y++)
          for (uint64 x = 0; x < size.x(); x++)
          {
            if (reaching_result.status.at<uint8>(y, x) != EvaluateViewFilter::ReachingResult::OCCUPIED)
              continue;

            predicted_mask.at<float>(y, x) = fixed_prob;
          }

        if (external_mask.data)
          predicted_mask = predicted_mask.mul(external_mask);
      }
      else
      {
        predicted_mask = 0.0f;
        for (uint64 y = 0; y < size.y(); y++)
          for (uint64 x = 0; x < size.x(); x++)
          {
            if (circle_rectangle_mask.at<uint8>(y, x))
              predicted_mask.at<float>(y, x) = fixed_prob;
          }

        if (external_mask.data)
          predicted_mask = predicted_mask.mul(external_mask);
      }
      cv_bridge->image = predicted_mask;
      cv_bridge->encoding = "32FC1";
      result.predicted_mask = *cv_bridge->toImageMsg();
    }

    {
      cv::Mat depth = cv::Mat(size.y(), size.x(), CV_32FC1, 0.0f);
      cv::Mat status = cv::Mat(size.y(), size.x(), CV_8UC1, uint8_t(0));
      cv::Mat normals = cv::Mat(size.y(), size.x(), CV_32FC3, cv::Vec3f(0.0f, 0.0f, 0.0f));
      if (maybe_reaching_result)
      {
        const EvaluateViewFilter::ReachingResult & reaching_result = *maybe_reaching_result;
        depth = reaching_result.depth;
        status = reaching_result.status;
        normals = reaching_result.normals;
      }
      cv_bridge->image = depth;
      cv_bridge->encoding = "32FC1";
      result.predicted_depth_image = *cv_bridge->toImageMsg();
      cv_bridge->image = status;
      cv_bridge->encoding = "8UC1";
      result.predicted_depth_image_status = *cv_bridge->toImageMsg();
      cv_bridge->image = normals;
      cv_bridge->encoding = "32FC3";
      result.predicted_normal_image = *cv_bridge->toImageMsg();
    }

    {
      cv_bridge->image = cv::Mat(size.y(), size.x(), CV_32FC1);
      for (uint64 y = 0; y < size.y(); y++)
        for (uint64 x = 0; x < size.x(); x++)
          cv_bridge->image.at<float>(y, x) = scores[x + y * size.x()];
      cv_bridge->encoding = "32FC1";
      result.predicted_score_image = *cv_bridge->toImageMsg();
    }

    {
      cv_bridge->image = cv::Mat(size.y(), size.x(), CV_32FC1);
      for (uint64 y = 0; y < size.y(); y++)
        for (uint64 x = 0; x < size.x(); x++)
          cv_bridge->image.at<float>(y, x) = unmasked_scores[x + y * size.x()];
      cv_bridge->encoding = "32FC1";
      result.predicted_unmasked_score_image = *cv_bridge->toImageMsg();
    }

    m_evaluate_view_action_server->setSucceeded(result);
    ROS_INFO("nbv_3d_cnn_evaluate_view: onEvaluateViewAction: end.");
  }

  void GetRobotDepthImage(const Eigen::Affine3d & camera_pose,
                          const sensor_msgs::JointState & joint_state,
                          const Eigen::Vector2f & center,
                          const Eigen::Vector2f & focal,
                          const Eigen::Vector2i & size,
                          const float range,
                          cv::Mat & depth_image,
                          cv::Mat & normal_image
                          )
  {
    ROS_INFO("nbv_3d_cnn_evaluate_view: waiting for render robot service client...");
    m_render_robot_service_client.waitForExistence();

    ROS_INFO("nbv_3d_cnn_evaluate_view: sending service call...");

    Eigen::Affine3d robot_pose = Eigen::Affine3d::Identity();

    render_robot_urdf::RenderRobotUrdf srv;
    srv.request.center_x = center.x();
    srv.request.center_y = center.y();
    srv.request.focal_x = focal.x();
    srv.request.focal_y = focal.y();
    srv.request.min_range = 0.1;
    srv.request.max_range = range;
    srv.request.width = size.x();
    srv.request.height = size.y();

    tf::poseEigenToMsg(camera_pose, srv.request.camera_pose);
    tf::poseEigenToMsg(robot_pose, srv.request.robot_pose);

    srv.request.joint_state = joint_state;

    if (m_render_robot_service_client.call(srv))
    {
      ROS_INFO("nbv_3d_cnn_evaluate_view: call succeeded!");

      {
        cv_bridge::CvImage cv_bridge = *cv_bridge::toCvCopy(srv.response.depth_image);
        depth_image = cv_bridge.image;
      }

      {
        cv_bridge::CvImage cv_bridge = *cv_bridge::toCvCopy(srv.response.normal_image);
        normal_image = cv_bridge.image;
      }
    }
    else
    {
      ROS_ERROR("nbv_3d_cnn_evaluate_view: call failed!");
    }
  }

  private:
  ros::NodeHandle & m_nh;

  std::mutex m_mutex;

  EnvironmentActionServerPtr m_environment_action_server;
  ViewActionServerPtr m_evaluate_view_action_server;
  RawProjectionActionServerPtr m_raw_projection_action_server;
  GenerateFilterCircleRectangleMaskServerPtr m_generate_filter_circle_rectangle_mask_server;
  GTViewActionServerPtr m_ground_truth_evaluate_view_action_server;

  EvaluateViewOpenCL m_opencl;

  EvaluateViewFilter m_view_filter;

  ros::ServiceClient m_render_robot_service_client;

  float m_voxel_size;
  Eigen::Vector3f m_environment_origin;
  float m_environment_origin_offset;

  float m_a_priori_ray_lost_if_occupied;
  float m_a_priori_ray_occupied_if_outside;
  float m_a_priori_occupancy_probability;

  uint64 m_cnn_image_depth;
  float m_cnn_max_range;

  Voxelgrid::Ptr m_current_probabilities;
  Voxelgrid::Ptr m_current_environment;
  Voxelgrid::Ptr m_current_cropped_environment;
  Vector4iVector m_current_tsdf;

  AutocompletePredict m_autocomplete_predict;
  ImagePredict m_image_predict;
  ProjectionImagePredict m_projection_image_predict;

  public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
};

int main(int argc, char ** argv)
{
  ros::init(argc, argv, "nbv_3d_cnn_evaluate_view");

  ros::NodeHandle nh("~");

  NBV3DCNNEvaluateView ncev(nh);

  ros::spin();

  return 0;
}
