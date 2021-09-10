#include "cnn_real_image_evaluate.h"

#include <ros/ros.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/JointState.h>
#include <actionlib/client/simple_action_client.h>
#include <message_serialization/serialize.h>
#include <message_serialization/sensor_msgs_yaml.h>
#include <eigen_conversions/eigen_msg.h>
#include <cv_bridge/cv_bridge.h>
#include <pcl_conversions/pcl_conversions.h>

#include <rmonica_voxelgrid_common/voxelgrid.h>
#include <rmonica_voxelgrid_common/metadata.h>
#include <nbv_3d_cnn_evaluate_view_msgs/EvaluateViewAction.h>
#include <nbv_3d_cnn_evaluate_view_msgs/SetEnvironmentAction.h>

#include <string>
#include <memory>
#include <sstream>
#include <stdint.h>
#include <fstream>

#include <Eigen/Dense>
#include <Eigen/StdVector>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>

class CNNRealImageEvaluate
{
  public:
  typedef uint64_t uint64;
  typedef uint8_t uint8;

  enum class Mode
  {
    NONE,
    FIXED,
    ADHOC,
    ONLY2D,
    FULL,
    GT
  };

  template <typename T>
  static T SQR(const T & t) {return t * t; }

  typedef rmonica_voxelgrid_common::Voxelgrid Voxelgrid;
  typedef VoxelgridMetadata::Metadata Metadata;
  typedef VoxelgridMetadata::MetadataPtr MetadataPtr;

  typedef actionlib::SimpleActionClient<nbv_3d_cnn_evaluate_view_msgs::SetEnvironmentAction> SetEnvironmentActionClient;
  typedef std::shared_ptr<SetEnvironmentActionClient> SetEnvironmentActionClientPtr;
  typedef actionlib::SimpleActionClient<nbv_3d_cnn_evaluate_view_msgs::EvaluateViewAction> EvaluateViewActionClient;
  typedef std::shared_ptr<EvaluateViewActionClient> EvaluateViewActionClientPtr;

  typedef pcl::PointCloud<pcl::PointXYZI> PointXYZICloud;

  Eigen::Vector3f StringToVector3f(const std::string & str)
  {
    Eigen::Vector3f result;
    std::istringstream istr(str);
    istr >> result.x() >> result.y() >> result.z();
    if (!istr)
    {
      ROS_FATAL("cnn_real_image_evaluate: unable to parse vector3f: %s", str.c_str());
      exit(1);
    }
    return result;
  }

  CNNRealImageEvaluate(ros::NodeHandle & nh): m_nh(nh)
  {
    std::string param_string;
    int param_int;
    double param_double;

    m_timer = m_nh.createTimer(ros::Duration(0.0001), &CNNRealImageEvaluate::onTimer, this, false);

    m_nh.param<std::string>(PARAM_NAME_MODE, param_string, PARAM_DEFAULT_MODE);
    if (param_string == PARAM_VALUE_MODE_NONE)
      m_mode = Mode::NONE;
    else if (param_string == PARAM_VALUE_MODE_FIXED)
      m_mode = Mode::FIXED;
    else if (param_string == PARAM_VALUE_MODE_ADHOC)
      m_mode = Mode::ADHOC;
    else if (param_string == PARAM_VALUE_MODE_ONLY2D)
      m_mode = Mode::ONLY2D;
    else if (param_string == PARAM_VALUE_MODE_FULL)
      m_mode = Mode::FULL;
    else if (param_string == PARAM_VALUE_MODE_GT)
      m_mode = Mode::GT;
    else
    {
      ROS_FATAL("cnn_real_image_evaluate: unknown mode: %s", param_string.c_str());
      exit(1);
    }

    m_nh.param<std::string>(PARAM_NAME_GT_FILE_PREFIX, m_gt_file_prefix, PARAM_DEFAULT_GT_FILE_PREFIX);
    m_nh.param<std::string>(PARAM_NAME_SCENARIO_FILE_PREFIX, m_scenario_file_prefix, PARAM_DEFAULT_SCENARIO_FILE_PREFIX);
    m_nh.param<std::string>(PARAM_NAME_VARIANT_FILE_PREFIX, m_environment_file_prefix, PARAM_DEFAULT_VARIANT_FILE_PREFIX);

    m_nh.param<int>(PARAM_NAME_SCENARIO_FIRST_INDEX, param_int, PARAM_DEFAULT_SCENARIO_FIRST_INDEX);
    m_scenario_first_index = param_int;
    m_nh.param<int>(PARAM_NAME_SCENARIO_LAST_INDEX, param_int, PARAM_DEFAULT_SCENARIO_LAST_INDEX);
    m_scenario_last_index = param_int;

    m_nh.param<bool>(PARAM_NAME_WITH_SALIENCY_IMAGES, m_with_saliency_images, PARAM_DEFAULT_WITH_SALIENCY_IMAGES);

    m_nh.param<std::string>(PARAM_NAME_EVALUATION_FILE_PREFIX, m_evaluation_file_prefix, PARAM_DEFAULT_EVALUATION_FILE_PREFIX);

    m_nh.param<std::string>(PARAM_NAME_IMAGE_FILE_PREFIX, m_image_file_prefix, PARAM_DEFAULT_IMAGE_FILE_PREFIX);

    m_nh.param<std::string>(PARAM_NAME_MASK_FILE_NAME, m_mask_file_name, PARAM_DEFAULT_MASK_FILE_NAME);

    m_nh.param<double>(PARAM_NAME_FIXED_MODE_FIXED_PROBABILITY, param_double, PARAM_DEFAULT_FIXED_MODE_FIXED_PROBABILITY);
    m_fixed_mode_fixed_probability = param_double;

    m_nh.param<bool>(PARAM_NAME_SAVE_IMAGES, m_save_images, PARAM_DEFAULT_SAVE_IMAGES);

    m_nh.param<std::string>(PARAM_NAME_CNN_BOUNDING_BOX_MAX, param_string, PARAM_DEFAULT_CNN_BOUNDING_BOX_MAX);
    m_cnn_bounding_box_max = StringToVector3f(param_string);

    m_nh.param<std::string>(PARAM_NAME_CNN_BOUNDING_BOX_MIN, param_string, PARAM_DEFAULT_CNN_BOUNDING_BOX_MIN);
    m_cnn_bounding_box_min = StringToVector3f(param_string);

    m_nh.param<double>(PARAM_NAME_CNN_BOUNDING_BOX_VOXEL_SIZE, param_double, PARAM_DEFAULT_CNN_BOUNDING_BOX_VOXEL_SIZE);
    m_cnn_bounding_box_voxel_size = param_double;

    m_set_environment_ac.reset(new SetEnvironmentActionClient("/evaluate_view/set_environment", true));
    m_evaluate_view_ac.reset(new EvaluateViewActionClient("/evaluate_view/evaluate_view", true));

    m_current_scenario = m_scenario_first_index;
    m_current_environment_variant = 0;
    m_current_image = 0;

    m_first_iter_scenario = true;
    m_mask_loaded = false;
    m_first_iter_environment_variant = false;
  }

  uint64 GetMaxExistingFile(const std::string & prefix, const std::string & suffix)
  {
    uint64 result = 0;
    const uint64 SANITY = 1000000;

    for (result = 0; result < SANITY; result++)
    {
      const std::string filename = prefix + std::to_string(result) + suffix;
      std::ifstream ifile(filename.c_str());
      if (!ifile)
        break;
    }

    return result;
  }

  Eigen::Affine3f LoadPose(const std::string pose_filename)
  {
    Eigen::Affine3f pose;
    ROS_INFO("cnn_real_image_evaluate: loading pose %s", pose_filename.c_str());
    std::ifstream ifile(pose_filename.c_str());
    for (uint64 y = 0; y < 3; y++)
    {
      for (uint64 x = 0; x < 3; x++)
        ifile >> pose.linear()(y, x);
      ifile >> pose.translation()[y];
    }
    if (!ifile)
    {
      ROS_ERROR("cnn_real_image_evaluate: load pose failed: %s", pose_filename.c_str());
      exit(1);
    }
    return pose;
  }

  sensor_msgs::JointState LoadJointState(const std::string joint_state_filename)
  {
    sensor_msgs::JointState joint_state;
    std::ifstream ifile(joint_state_filename.c_str());
    decltype(joint_state.position) js;
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

    return joint_state;
  }

  // x, y, z, radius
  Eigen::Vector4f LoadPOI(const std::string poi_filename)
  {
    Eigen::Vector4f poi;
    std::ifstream ifile(poi_filename.c_str());
    ifile >> poi.x() >> poi.y() >> poi.z() >> poi.w();
    if (!ifile)
    {
      ROS_ERROR("cnn_real_image_evaluate: could not load poi %s", poi_filename.c_str());
      exit(1);
    }
    return poi;
  }

  sensor_msgs::CameraInfo LoadCameraInfo(const std::string & filename)
  {
    sensor_msgs::CameraInfo camera_info;
    if (!message_serialization::deserialize(filename, camera_info))
    {
      ROS_ERROR("cnn_real_image_evaluate: could not load camera info %s!", filename.c_str());
      exit(1);
    }

    return camera_info;
  }

  struct Prediction
  {
    cv::Mat mask;
    cv::Mat scores;
    cv::Mat unmasked_scores;
    float gain = 0.0f;

    double network_prediction_time;
    double raycast_time;
    double robot_time;
    double filter_time;
    double prob_raycast_time;
    double total_time;

    std::vector<cv::Mat> saliency_images;
  };

  void LoadVoxelgrid(const std::string & voxelgrid_filename,
                     const std::string & metadata_filename,
                     const std::string & tsdf_volume_filename)
  {
    ROS_INFO("cnn_real_image_evaluate: loading voxelgrid from file %s", voxelgrid_filename.c_str());
    const Voxelgrid::Ptr vg = Voxelgrid::FromFileTernaryBinary(voxelgrid_filename);
    if (!vg)
    {
      ROS_ERROR("cnn_real_image_evaluate: could not load voxelgrid.");
      exit(1);
    }

    const Eigen::Vector3i voxelgrid_size = vg->GetSize();

    ROS_INFO("cnn_real_image_evaluate: loading voxelgrid metadata from file %s", metadata_filename.c_str());
    MetadataPtr metadata = VoxelgridMetadata::LoadMetadata(metadata_filename);
    if (!metadata)
    {
      ROS_ERROR("cnn_real_image_evaluate: could not load grid metadata!");
      exit(1);
    }

    ROS_INFO_STREAM("cnn_real_image_evaluate: bounding box is " << metadata->bbox_max.transpose() << " - "
                    << metadata->bbox_max.transpose());
    ROS_INFO_STREAM("cnn_real_image_evaluate: voxel size is " << metadata->voxel_size);

    ROS_INFO("cnn_real_image_evaluate: loading TSDF from file %s", tsdf_volume_filename.c_str());
    PointXYZICloud tsdf_cloud;
    if (pcl::io::loadPCDFile(tsdf_volume_filename, tsdf_cloud) < 0)
    {
      ROS_ERROR("cnn_real_image_evaluate: could not load TSDF cloud!");
      exit(1);
    }
    ROS_INFO("cnn_real_image_evaluate: TSDF cloud has size %u", unsigned(tsdf_cloud.size()));

    Voxelgrid vg2 = *vg;

    Voxelgrid prob_vg;
    const Eigen::Vector3i probabilities_origin = ((m_cnn_bounding_box_min - metadata->bbox_min) /
                                                  metadata->voxel_size).array().round().cast<int>();
    const Eigen::Vector3i probabilities_size = ((m_cnn_bounding_box_max - m_cnn_bounding_box_min) /
                                                metadata->voxel_size).array().round().cast<int>();
    {
      nbv_3d_cnn_evaluate_view_msgs::SetEnvironmentGoal se_goal;
      se_goal.voxel_size = metadata->voxel_size;

      se_goal.environment_origin.x = metadata->bbox_min.x();
      se_goal.environment_origin.y = metadata->bbox_min.y();
      se_goal.environment_origin.z = metadata->bbox_min.z();

      se_goal.environment = vg2.ToInt8MultiArray();

      se_goal.probabilities_voxel_size = m_cnn_bounding_box_voxel_size * metadata->voxel_size;

      se_goal.probabilities_origin[0] = probabilities_origin.x();
      se_goal.probabilities_origin[1] = probabilities_origin.y();
      se_goal.probabilities_origin[2] = probabilities_origin.z();
      se_goal.probabilities_size[0] = probabilities_size.x();
      se_goal.probabilities_size[1] = probabilities_size.y();
      se_goal.probabilities_size[2] = probabilities_size.z();

      pcl::toROSMsg(tsdf_cloud, se_goal.tsdf_cloud);

      ROS_INFO("cnn_real_image_evaluate: setting environment...");
      m_set_environment_ac->waitForServer();
      m_set_environment_ac->sendGoal(se_goal);
      m_set_environment_ac->waitForResult();

      nbv_3d_cnn_evaluate_view_msgs::SetEnvironmentResultConstPtr se_result = m_set_environment_ac->getResult();
      prob_vg = *Voxelgrid::FromFloat32MultiArray(se_result->predicted_probabilities);
    }
  }

  Prediction CallEvaluateView(const sensor_msgs::CameraInfo & camera_info,
                              const sensor_msgs::JointState & joint_state,
                              const Eigen::Vector3f & poi,
                              const float poi_radius,
                              const Eigen::Affine3f & pose,
                              const Mode mode,
                              const cv::Mat external_mask
                              )
  {
    nbv_3d_cnn_evaluate_view_msgs::EvaluateViewGoal ev_goal;
    ev_goal.center_x = camera_info.K[2];
    ev_goal.center_y = camera_info.K[5];
    ev_goal.focal_x = camera_info.K[0];
    ev_goal.focal_y = camera_info.K[4];
    ev_goal.size_x = camera_info.width;
    ev_goal.size_y = camera_info.height;

    ev_goal.joint_state = joint_state;

    ev_goal.max_range = 4.0;
    ev_goal.min_range = 0.5;

    ev_goal.has_roi_sphere = true;
    ev_goal.roi_sphere_center.x = poi.x();
    ev_goal.roi_sphere_center.y = poi.y();
    ev_goal.roi_sphere_center.z = poi.z();
    ev_goal.roi_sphere_radius = poi_radius;

    ev_goal.with_saliency_images = m_with_saliency_images;

    if (mode == Mode::ADHOC)
      ev_goal.mode = ev_goal.MODE_ADVANCED_PROB;
    else if (mode == Mode::NONE || mode == Mode::GT)
      ev_goal.mode = ev_goal.MODE_STANDARD_PROB;
    else if (mode == Mode::FIXED)
      ev_goal.mode = ev_goal.MODE_FIXED;
    else if (mode == Mode::ONLY2D)
      ev_goal.mode = ev_goal.MODE_IMAGE_PREDICT_PROB;
    else if (mode == Mode::FULL)
      ev_goal.mode = ev_goal.MODE_PROJ_PREDICT_PROB;
    else
    {
      ROS_FATAL("cnn_real_image_evaluate: unknown mode %u", unsigned(mode));
      exit(1);
    }

    {
      cv_bridge::CvImagePtr cv_bridge(new cv_bridge::CvImage);
      cv_bridge->image = external_mask;
      cv_bridge->encoding = "32FC1";
      ev_goal.external_mask = *cv_bridge->toImageMsg();
    }

    ev_goal.fixed_probability = m_fixed_mode_fixed_probability;

    tf::poseEigenToMsg(pose.cast<double>(), ev_goal.pose);

    ROS_INFO("cnn_real_image_evaluate: sending evaluation goal...");
    m_evaluate_view_ac->sendGoal(ev_goal);
    m_evaluate_view_ac->waitForResult();

    nbv_3d_cnn_evaluate_view_msgs::EvaluateViewResultConstPtr ev_result = m_evaluate_view_ac->getResult();

    Prediction result;

    cv_bridge::CvImagePtr cv_bridge;
    cv_bridge = cv_bridge::toCvCopy(ev_result->predicted_mask);
    if (!cv_bridge || !cv_bridge->image.data)
    {
      ROS_ERROR("cnn_real_image_evaluate: could not read predicted mask!");
      return result;
    }
    const cv::Mat mask = cv_bridge->image;
    result.mask = mask;

    {
      cv_bridge::CvImagePtr cv_bridge;
      cv_bridge = cv_bridge::toCvCopy(ev_result->predicted_score_image);
      if (!cv_bridge || !cv_bridge->image.data)
      {
        ROS_ERROR("cnn_real_image_evaluate: could not read predicted scores!");
        return result;
      }
      result.scores = cv_bridge->image;
    }

    {
      cv_bridge::CvImagePtr cv_bridge;
      cv_bridge = cv_bridge::toCvCopy(ev_result->predicted_unmasked_score_image);
      if (!cv_bridge || !cv_bridge->image.data)
      {
        ROS_ERROR("cnn_real_image_evaluate: could not read predicted unmasked scores!");
        return result;
      }
      result.unmasked_scores = cv_bridge->image;
    }

    result.gain = ev_result->gain;

    {
      cv_bridge::CvImagePtr cv_bridge;
      for (const sensor_msgs::Image & img_msg : ev_result->saliency_images)
      {
        cv_bridge = cv_bridge::toCvCopy(img_msg);
        if (!cv_bridge || !cv_bridge->image.data)
        {
          ROS_ERROR("cnn_real_image_evaluate: could not read saliency image!");
          return result;
        }
        result.saliency_images.push_back(cv_bridge->image);
      }
    }

    result.network_prediction_time = ev_result->network_prediction_time;
    result.raycast_time = ev_result->raycast_time;
    result.robot_time = ev_result->robot_time;
    result.filter_time = ev_result->filter_time;
    result.prob_raycast_time = ev_result->prob_raycast_time;
    result.total_time = ev_result->total_time;

    return result;
  }

  Prediction PredictImage(const uint64 scenario_index,
                          const uint64 environment_variant_index,
                          const uint64 image_index,
                          const cv::Mat mask)
  {
    static uint64 prev_environment_variant_index = -1;
    static uint64 prev_image_index = -1;

    const std::string scenario_str = std::to_string(scenario_index);
    const std::string image_str = std::to_string(image_index);
    const std::string environment_variant_str = std::to_string(environment_variant_index);

    sensor_msgs::CameraInfo camera_info;
    sensor_msgs::JointState joint_state;
    Eigen::Affine3f pose;
    Eigen::Vector3f poi;
    float poi_radius;

    {
      const std::string camera_info_filename = m_scenario_file_prefix + scenario_str + m_image_file_prefix +
                                               "camera_info_" + image_str + ".txt";
      camera_info = LoadCameraInfo(camera_info_filename);
    }

    {
      const std::string joint_state_filename = m_scenario_file_prefix + scenario_str + m_image_file_prefix +
                                               "joint_state_" + image_str + ".txt";
      joint_state = LoadJointState(joint_state_filename);
    }

    {
      const std::string pose_filename = m_scenario_file_prefix + scenario_str + m_image_file_prefix +
                                               "pose_" + image_str + ".matrix";
      pose = LoadPose(pose_filename);
    }

    {
      const std::string poi_filename = m_scenario_file_prefix + scenario_str + "/poi.txt";
      const Eigen::Vector4f poi4f = LoadPOI(poi_filename);
      poi = poi4f.head<3>();
      poi_radius = poi4f[3];
    }

    return CallEvaluateView(camera_info, joint_state, poi, poi_radius, pose, m_mode, mask);
  }

  void CompareScoreImages(const cv::Mat & image,
                          const cv::Mat & ground_truth,
                          const cv::Mat & mask,
                          float &rmse_error
                          )
  {
    const uint64 width = image.cols;
    const uint64 height = image.rows;

    uint64 counter = 0;
    rmse_error = 0.0f;
    for (uint64 y = 0; y < height; y++)
      for (uint64 x = 0; x < width; x++)
      {
        if (!mask.at<float>(y, x))
          continue;
        counter++;
        const float diff = image.at<float>(y, x) - ground_truth.at<float>(y, x);
        rmse_error += SQR(diff);
      }

    rmse_error /= float(counter);
    rmse_error = std::sqrt(rmse_error);
  }

  void UpdateEvaluationFile(const uint64 scenario,
                            const uint64 variant,
                            const uint64 image,
                            const Prediction & prediction)
  {
    if (!m_evaluation_file)
    {
      const std::string log_file_name = m_evaluation_file_prefix + "aaa_log.csv";
      ROS_INFO("cnn_real_image_evaluate: saving to log file %s", log_file_name.c_str());
      m_evaluation_file.reset(new std::ofstream(log_file_name.c_str()));
      if (!*m_evaluation_file)
      {
        ROS_FATAL("cnn_real_image_evaluate: could not create evaluation file %s", log_file_name.c_str());
        exit(1);
      }

      (*m_evaluation_file) << "Scn" << "\t" << "Image" << "\t" << "Variant" << "\t"
                           << "Gain" << "\n";
    }

    ROS_INFO("cnn_real_image_evaluate: updating evaluation file.");
    (*m_evaluation_file) << scenario << "\t" << image << "\t" << variant << "\t"
                         << prediction.gain << "\n";
    if (image % 50 == 0)
      (*m_evaluation_file) << std::flush;
  }

  void UpdateTimersFile(const uint64 scenario,
                        const uint64 variant,
                        const uint64 image,
                        const Prediction & prediction)
  {
    if (!m_timers_file)
    {
      const std::string log_file_name = m_evaluation_file_prefix + "aaa_timers.csv";
      ROS_INFO("cnn_real_image_evaluate: saving to timers file %s", log_file_name.c_str());
      m_timers_file.reset(new std::ofstream(log_file_name.c_str()));
      if (!*m_timers_file)
      {
        ROS_FATAL("cnn_real_image_evaluate: could not create timers file %s", log_file_name.c_str());
        exit(1);
      }

      (*m_timers_file) << "Scn" << "\t" << "Image" << "\t" << "Variant" << "\t"
                           << "NetPred" << "\t" << "Raycast" << "\t" << "Robot" << "\t" << "Filter"
                           << "\t" << "ProbRaycast" << "\t" << "Total" <<  "\n";
    }

    ROS_INFO("cnn_real_image_evaluate: updating timers file.");
    (*m_timers_file) << scenario << "\t" << image << "\t" << variant << "\t"
                     << prediction.network_prediction_time << "\t" << prediction.raycast_time << "\t"
                     << prediction.robot_time << "\t" << prediction.filter_time
                     << "\t" << prediction.prob_raycast_time << "\t" << prediction.total_time <<"\n";
    if (image % 50 == 0)
      (*m_timers_file) << std::flush;
  }

  void onTimer(const ros::TimerEvent &)
  {
    ROS_INFO("cnn_real_image_evaluate: processin start, scenario %u, image %u, variant %u.",
             unsigned(m_current_scenario), unsigned(m_current_image), unsigned(m_current_environment_variant));

    if (!m_mask_loaded)
    {
      ROS_INFO("cnn_real_image_evaluate: loading mask %s.", m_mask_file_name.c_str());
      m_mask = cv::imread(m_mask_file_name, cv::IMREAD_GRAYSCALE);
      if (!m_mask.data)
      {
        ROS_FATAL("cnn_real_image_evaluate: could not load mask %s.", m_mask_file_name.c_str());
        exit(1);
      }
      m_mask.convertTo(m_mask, CV_32FC1);
      m_mask_loaded = true;
    }

    const std::string scenario_str = std::to_string(m_current_scenario);

    if (m_first_iter_scenario)
    {
      ROS_INFO("cnn_real_image_evaluate: switching to scenario %u", unsigned(m_current_scenario));
      const std::string image_prefix = m_scenario_file_prefix + scenario_str + m_gt_file_prefix + "gt_";
      const std::string image_suffix = ".png";
      m_max_images = GetMaxExistingFile(image_prefix, image_suffix);
      ROS_INFO("cnn_real_image_evaluate: scenario is %u, found %u images.",
               unsigned(m_current_scenario), unsigned(m_max_images));

      const std::string variant_prefix = m_scenario_file_prefix + scenario_str + m_environment_file_prefix + "partial_";
      const std::string variant_suffix = ".voxelgrid";
      m_max_environment_variants = GetMaxExistingFile(variant_prefix, variant_suffix);
      ROS_INFO("cnn_real_image_evaluate: scenario is %u, found %u environment variants.",
               unsigned(m_current_scenario), unsigned(m_max_environment_variants));

      m_first_iter_scenario = false;
      m_current_image = 0;
      m_current_environment_variant = 0;
      m_first_iter_environment_variant = true;
    }

    const std::string environment_variant_str = std::to_string(m_current_environment_variant);

    if (m_first_iter_environment_variant)
    {
      ROS_INFO("cnn_real_image_evaluate: switching to environment variant %u", unsigned(m_current_environment_variant));

      const std::string voxelgrid_filename = m_scenario_file_prefix + scenario_str +
          m_environment_file_prefix + "partial_" + environment_variant_str + ".voxelgrid";
      const std::string voxelgrid_metadata_filename = m_scenario_file_prefix + scenario_str + "/voxelgrid_metadata.txt";
      const std::string tsdf_volume_filename = m_scenario_file_prefix + scenario_str + "/kinfu_tsdf.pcd";

      LoadVoxelgrid(voxelgrid_filename, voxelgrid_metadata_filename, tsdf_volume_filename);

      m_first_iter_environment_variant = false;
      m_current_image = 0;
    }

    const std::string image_str = std::to_string(m_current_image);

    cv::Mat mask = m_mask.clone();
    if (m_mode == Mode::GT)
    {
      const std::string gt_filename = m_scenario_file_prefix + scenario_str +
                                      m_gt_file_prefix + "gt_" + image_str + ".png";
      ROS_INFO("cnn_real_image_evaluate: loading ground truth image %s", gt_filename.c_str());
      cv::Mat gt_image = cv::imread(gt_filename, cv::IMREAD_GRAYSCALE);
      if (!gt_image.data)
      {
        ROS_FATAL("cnn_real_image_evaluate: could not load image %s", gt_filename.c_str());
        exit(1);
      }
      gt_image.convertTo(gt_image, CV_32FC1, 1.0 / 255.0);

      mask = mask.mul(gt_image);
    }

    Prediction prediction = PredictImage(m_current_scenario,
                                         m_current_environment_variant,
                                         m_current_image,
                                         mask);

    ROS_INFO("cnn_real_image_evaluate: gain is %f", float(prediction.gain));

    UpdateEvaluationFile(m_current_scenario, m_current_environment_variant, m_current_image, prediction);
    UpdateTimersFile(m_current_scenario, m_current_environment_variant, m_current_image, prediction);


    if (m_save_images)
    {
      {
        const std::string mask_filename = m_evaluation_file_prefix + "mask_" +
                                          scenario_str + "_" + image_str + "_" + environment_variant_str + ".png";
        ROS_INFO("cnn_real_image_evaluate: saving mask to file %s.", mask_filename.c_str());
        cv::Mat int_mask;
        prediction.mask.convertTo(int_mask, CV_8UC1, 255.0f);
        cv::imwrite(mask_filename, int_mask);
      }

      {
        const std::string score_filename = m_evaluation_file_prefix + "score_" +
                                           scenario_str + "_" + image_str + "_" + environment_variant_str + ".png";
        ROS_INFO("cnn_real_image_evaluate: saving scores to file %s.", score_filename.c_str());
        cv::Mat int_scores;
        prediction.scores.convertTo(int_scores, CV_16UC1, 10.0 * 1000.0);
        cv::imwrite(score_filename, int_scores);
      }

      {
        const std::string unmasked_score_filename = m_evaluation_file_prefix + "unmasked_score_" +
                                           scenario_str + "_" + image_str + "_" + environment_variant_str + ".png";
        ROS_INFO("cnn_real_image_evaluate: saving unmasked scores to file %s.", unmasked_score_filename.c_str());
        cv::Mat int_scores;
        prediction.unmasked_scores.convertTo(int_scores, CV_16UC1, 10.0 * 1000.0);
        cv::imwrite(unmasked_score_filename, int_scores);
      }

      if (!prediction.saliency_images.empty())
      {
        const std::string saliency_filename = m_evaluation_file_prefix + "saliency_" +
                                             scenario_str + "_" + image_str + "_" + environment_variant_str + ".png";
        cv::Mat concat_image;
        cv::hconcat(prediction.saliency_images.data(), prediction.saliency_images.size(), concat_image);

        ROS_INFO("cnn_real_image_evaluate: saving saliency to file %s.", saliency_filename.c_str());
        {
          cv::Mat int_saliency;
          concat_image.convertTo(int_saliency, CV_16UC1, 1000.0);
          cv::imwrite(saliency_filename, int_saliency);
        }
      }
    }

    m_current_image++;
    if (m_current_image >= m_max_images)
    {
      m_current_environment_variant++;
      ROS_INFO("cnn_real_image_evaluate: current env. variant is now %u", unsigned(m_current_environment_variant));
      m_current_image = 0;
      m_first_iter_environment_variant = true;
    }

    if (m_current_environment_variant >= m_max_environment_variants)
    {
      m_current_scenario++;
      m_first_iter_scenario = true;
    }

    if (m_current_scenario >= m_scenario_last_index)
    {
      ROS_INFO("cnn_real_image_evaluate: all scenarios processed, end.");
      m_timer.stop();

      if (m_evaluation_file)
        m_evaluation_file.reset();
      if (m_timers_file)
        m_timers_file.reset();
    }
  }

  private:
  ros::NodeHandle & m_nh;

  ros::Timer m_timer;

  uint64 m_current_scenario;
  bool m_first_iter_scenario;
  uint64 m_current_environment_variant;
  bool m_first_iter_environment_variant;
  uint64 m_current_image;

  uint64 m_max_images;
  uint64 m_max_environment_variants;

  std::string m_mask_file_name;
  bool m_mask_loaded;
  cv::Mat m_mask;

  Eigen::Vector3f m_cnn_bounding_box_min;
  Eigen::Vector3f m_cnn_bounding_box_max;
  float m_cnn_bounding_box_voxel_size;

  bool m_with_saliency_images;
  bool m_save_images;

  float m_fixed_mode_fixed_probability;

  uint64 m_scenario_first_index;
  uint64 m_scenario_last_index;
  std::string m_scenario_file_prefix;

  std::string m_gt_file_prefix;
  std::string m_environment_file_prefix;
  std::string m_image_file_prefix;

  SetEnvironmentActionClientPtr m_set_environment_ac;
  EvaluateViewActionClientPtr m_evaluate_view_ac;

  std::string m_evaluation_file_prefix;
  std::shared_ptr<std::ofstream> m_evaluation_file;
  std::shared_ptr<std::ofstream> m_timers_file;

  Mode m_mode;
};


int main(int argc, char ** argv)
{
  ros::init(argc, argv, "cnn_real_image_evaluate");
  ros::NodeHandle nh("~");

  CNNRealImageEvaluate crie(nh);

  ros::spin();

  return 0;
}
