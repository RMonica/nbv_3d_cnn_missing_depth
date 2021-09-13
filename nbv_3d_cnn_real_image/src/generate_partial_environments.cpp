#include "generate_partial_environments.h"

#include <ros/ros.h>
#include <std_msgs/Float32MultiArray.h>
#include <sensor_msgs/CameraInfo.h>

#include <vector>
#include <string>
#include <sstream>
#include <stdint.h>

#include <Eigen/Dense>
#include <Eigen/StdVector>

#include <rmonica_voxelgrid_common/voxelgrid.h>
#include <rmonica_voxelgrid_common/metadata.h>

// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

// Serialization headers
#include <message_serialization/serialize.h>
// Datatype-specific serialization header
#include <message_serialization/sensor_msgs_yaml.h>

using namespace VoxelgridMetadata;

class GeneratePartialEnvironments
{
  public:
  typedef uint64_t uint64;
  typedef uint16_t uint16;
  typedef std::vector<uint64> Uint64Vector;
  typedef rmonica_voxelgrid_common::Voxelgrid Voxelgrid;

  struct Viewpoint
  {
    Eigen::Affine3f pose;
    Eigen::Affine3f pose_inv;
    float fx, fy, cx, cy;
    uint64 width, height;

    cv::Mat depth_image;
  };

  typedef std::vector<Viewpoint, Eigen::aligned_allocator<Viewpoint> > ViewpointVector;

  GeneratePartialEnvironments(ros::NodeHandle & nh): m_nh(nh)
  {
    double param_double;
    int param_int;
    std::string param_string;

    m_nh.param<std::string>(PARAM_NAME_VOXELGRID_FILENAME, m_voxelgrid_filename, PARAM_DEFAULT_VOXELGRID_FILENAME);
    m_nh.param<std::string>(PARAM_NAME_METADATA_FILENAME, m_metadata_filename, PARAM_DEFAULT_METADATA_FILENAME);

    m_nh.param<std::string>(PARAM_NAME_OUTPUT_PREFIX, m_output_prefix, PARAM_DEFAULT_OUTPUT_PREFIX);
    m_nh.param<std::string>(PARAM_NAME_IMAGES_PREFIX, m_images_prefix, PARAM_DEFAULT_IMAGES_PREFIX);

    m_nh.param<std::string>(PARAM_NAME_CROP_BBOX_MIN, param_string, PARAM_DEFAULT_CROP_BBOX_MIN);
    {
      std::istringstream istr(param_string);
      istr >> m_crop_bbox_min.x() >> m_crop_bbox_min.y() >> m_crop_bbox_min.z();
      if (!istr)
      {
        ROS_FATAL("generate_partial_environments: unable to parse bbox min: %s", param_string.c_str());
        exit(1);
      }
    }

    m_nh.param<std::string>(PARAM_NAME_CROP_BBOX_MAX, param_string, PARAM_DEFAULT_CROP_BBOX_MAX);
    {
      std::istringstream istr(param_string);
      istr >> m_crop_bbox_max.x() >> m_crop_bbox_max.y() >> m_crop_bbox_max.z();
      if (!istr)
      {
        ROS_FATAL("generate_partial_environments: unable to parse bbox max: %s", param_string.c_str());
        exit(1);
      }
    }

    m_nh.param<std::string>(PARAM_NAME_POI_FILE_NAME, param_string, PARAM_DEFAULT_POI_FILE_NAME);
    {
      ROS_INFO("generate_partial_environments: loading POI from file: %s", param_string.c_str());
      std::ifstream ifile(param_string);
      ifile >> m_poi_center.x() >> m_poi_center.y() >> m_poi_center.z() >> m_poi_radius;
      if (!ifile)
      {
        ROS_FATAL("generate_partial_environments: unable to parse POI center: %s", param_string.c_str());
        exit(1);
      }
      ROS_INFO_STREAM("generate_partial_environments: POI is " << m_poi_center.transpose() <<
                      " radius " << m_poi_radius);
    }

    m_nh.param<int>(PARAM_NAME_SELECT_VIEWPOINTS_MIN, param_int, PARAM_DEFAULT_SELECT_VIEWPOINTS_MIN);
    m_select_viewpoints_min = param_int;

    m_nh.param<int>(PARAM_NAME_SELECT_VIEWPOINTS_MAX, param_int, PARAM_DEFAULT_SELECT_VIEWPOINTS_MAX);
    m_select_viewpoints_max = param_int;

    m_nh.param<int>(PARAM_NAME_NUM_INCOMPLETE_ENV, param_int, PARAM_DEFAULT_NUM_INCOMPLETE_ENV);
    m_num_incomplete_environments = param_int;

    m_timer = m_nh.createTimer(ros::Duration(0.5), &GeneratePartialEnvironments::onTimer, this, true);
  }

  void onTimer(const ros::TimerEvent&)
  {
    Generate();
  }

  bool CheckViewpoint(const uint64 number, const uint64 sub_number)
  {
    const std::string number_str = std::to_string(number);
    const std::string sub_number_str = std::to_string(sub_number);
    const std::string filename = m_images_prefix + "depth_image_" + number_str + "_" + sub_number_str + ".png";
    std::ifstream ifile(filename);
    if (ifile)
      return true;
    else
      return false;
  }

  Eigen::Affine3f LoadAffine3f(const std::string & filename)
  {
    Eigen::Affine3f result;
    ROS_INFO("generate_partial_environments: loading matrix %s", filename.c_str());
    std::ifstream ifile(filename);
    for (uint64 y = 0; y < 3; y++)
      for (uint64 x = 0; x < 4; x++)
      {
        float v;
        ifile >> v;
        result.matrix()(y, x) = v;
      }

    if (!ifile)
    {
      ROS_FATAL("generate_partial_environments: could not load matrix %s", filename.c_str());
      exit(1);
    }

    return result;
  }

  Viewpoint LoadViewpoint(const uint64 number)
  {
    cv::Mat total_depth_image;
    cv::Mat counters;
    const std::string number_str = std::to_string(number);
    for (uint64 sub_viewpoint_number = 0; CheckViewpoint(number, sub_viewpoint_number); sub_viewpoint_number++)
    {
      const std::string sub_number_str = std::to_string(sub_viewpoint_number);
      const std::string depth_filename = m_images_prefix + "depth_image_" + number_str + "_" + sub_number_str + ".png";
      cv::Mat this_depth_image = cv::imread(depth_filename, cv::IMREAD_ANYDEPTH);
      if (!this_depth_image.data)
      {
        ROS_FATAL("generate_partial_environments: could not load image %s", depth_filename.c_str());
        exit(1);
      }
      this_depth_image.convertTo(this_depth_image, CV_32FC1, 1 / 1000.0);

      const uint64 width = this_depth_image.cols;
      const uint64 height = this_depth_image.rows;

      if (sub_viewpoint_number == 0)
      {
        total_depth_image = cv::Mat(height, width, CV_32FC1, 0.0f);
        counters = cv::Mat(height, width, CV_32FC1, 0.0f);
      }

      for (uint64 y = 0; y < height; y++)
        for (uint64 x = 0; x < width; x++)
        {
          if (this_depth_image.at<float>(y, x) != 0.0f)
          {
            total_depth_image.at<float>(y, x) += this_depth_image.at<float>(y, x);
            counters.at<float>(y, x) += 1.0f;
          }
        }
    }
    if (!total_depth_image.data)
    {
      ROS_FATAL("generate_partial_environments: could not load any image for viewpoint %u", unsigned(number));
      exit(1);
    }

    cv::max(counters, 1.0f, counters);
    total_depth_image = total_depth_image / counters;

    const uint64 width = total_depth_image.cols;
    const uint64 height = total_depth_image.rows;

    Viewpoint result;
    result.depth_image = total_depth_image;
    result.height = height;
    result.width = width;

    const std::string camera_info_filename = m_images_prefix + "camera_info_" + number_str + ".txt";
    ROS_INFO("generate_partial_environments: loading camera info: %s", camera_info_filename.c_str());
    sensor_msgs::CameraInfo camera_info;
    if (!message_serialization::deserialize(camera_info_filename, camera_info))
    {
      ROS_FATAL("generate_partial_environments: could not open or deserialize file!");
      exit(1);
    }

    result.fx = camera_info.K[0];
    result.cx = camera_info.K[2];
    result.fy = camera_info.K[4];
    result.cy = camera_info.K[5];

    const std::string pose_filename = m_images_prefix + "pose_" + number_str + ".matrix";
    const Eigen::Affine3f pose = LoadAffine3f(pose_filename);
    result.pose = pose;
    result.pose_inv = pose.inverse();

    return result;
  }

  Voxelgrid ClearPOIExceptViewpoints(const Voxelgrid & voxelgrid,
                                     const Metadata & metadata,
                                     const Eigen::Vector3f & poi,
                                     const float poi_radius,
                                     const ViewpointVector & viewpoints)
  {
    const uint64 width = voxelgrid.GetWidth();
    const uint64 height = voxelgrid.GetHeight();
    const uint64 depth = voxelgrid.GetDepth();

    Voxelgrid result = voxelgrid;

    for (uint64 z = 0; z < depth; z++)
      for (uint64 y = 0; y < height; y++)
        for (uint64 x = 0; x < width; x++)
        {
          const Eigen::Vector3i ipt(x, y, z);
          const Eigen::Vector3f pt = (ipt.cast<float>() + Eigen::Vector3f::Ones() * 0.5) * metadata.voxel_size + metadata.bbox_min;

          if ((pt - poi).squaredNorm() > poi_radius * poi_radius)
            continue;

          bool in_vp = false;
          for (const Viewpoint & vp : viewpoints)
          {
            const Eigen::Vector3f local_pt = vp.pose_inv * pt;

            if (local_pt.z() <= 0.0f)
              continue;

            Eigen::Vector2i camera_pt;
            camera_pt.x() = std::round(local_pt.x() / local_pt.z() * vp.fx + vp.cx);
            camera_pt.y() = std::round(local_pt.y() / local_pt.z() * vp.fy + vp.cy);

            if ((camera_pt.array() < 0).any())
              continue;
            if ((camera_pt.array() >= Eigen::Vector2i(vp.width, vp.height).array()).any())
              continue;

            const float depth = vp.depth_image.at<float>(camera_pt.y(), camera_pt.x());
            if (local_pt.z() < depth + metadata.voxel_size)
              in_vp = true;

            if (in_vp)
              break;
          }

          if (!in_vp)
            result.at(x, y, z) = 0.0f;
        }

    return result;
  }

  void Generate()
  {
    ROS_INFO("generate_partial_environments: loading environment %s", m_voxelgrid_filename.c_str());
    Voxelgrid::Ptr environment_ptr = Voxelgrid::FromFileTernaryBinary(m_voxelgrid_filename);
    if (!environment_ptr)
    {
      ROS_ERROR("generate_partial_environments: could not load environment");
      return;
    }

    const Voxelgrid voxelgrid = *environment_ptr;

    ROS_INFO_STREAM("generate_partial_environments: environment size is " << voxelgrid.GetSize().transpose());

    ROS_INFO("generate_partial_environments: loading metadata %s", m_metadata_filename.c_str());
    const MetadataPtr metadata = LoadMetadata(m_metadata_filename);
    if (!metadata)
    {
      ROS_ERROR("generate_partial_environments: could not load metadata.");
      return;
    }

    ROS_INFO("generate_partial_environments: checking viewpoint existence.");
    uint64 max_available_viewpoints;
    for (max_available_viewpoints = 0; CheckViewpoint(max_available_viewpoints, 0); max_available_viewpoints++)
      ROS_INFO("generate_partial_environments: viewpoint %u exists.", unsigned(max_available_viewpoints));
    ROS_INFO("generate_partial_environments: detected %u viewpoints.", unsigned(max_available_viewpoints));

    for (uint64 incompl_i = 0; incompl_i < m_num_incomplete_environments + 2; incompl_i++)
    {
      ROS_INFO("generate_partial_environments: generate incomplete environment %u", unsigned(incompl_i));
      Uint64Vector candidates;
      candidates.reserve(max_available_viewpoints);
      for (uint64 i = 0; i < max_available_viewpoints; i++)
        candidates.push_back(i);

      Uint64Vector selected;
      uint64 num_candidates_to_select = rand() % (m_select_viewpoints_max - m_select_viewpoints_min + 1) + m_select_viewpoints_min;
      if (num_candidates_to_select > candidates.size())
        num_candidates_to_select = candidates.size();

      for (uint64 i = 0; i < num_candidates_to_select; i++)
      {
        const uint64 s = rand() % candidates.size();
        selected.push_back(candidates[s]);
        candidates.erase(candidates.begin() + s);
      }

      ROS_INFO("generate_partial_environments: selected candidates: ");
      for (uint64 i = 0; i < selected.size(); i++)
        ROS_INFO("generate_partial_environments:    %u", unsigned(selected[i]));

      ViewpointVector viewpoints;
      for (uint64 i = 0; i < selected.size(); i++)
      {
        const Viewpoint vp = LoadViewpoint(selected[i]);
        viewpoints.push_back(vp);
      }

      Voxelgrid cleared;

      if (incompl_i == 0)
      {
        cleared = voxelgrid; // do nothing (clear none)
      }
      else if (incompl_i == 1)
      {
        cleared = ClearPOIExceptViewpoints(voxelgrid, *metadata, m_poi_center, m_poi_radius, ViewpointVector()); // clear everything
      }
      else
      {
        cleared = ClearPOIExceptViewpoints(voxelgrid, *metadata, m_poi_center, m_poi_radius, viewpoints);
      }

      {
        const std::string output_filename = m_output_prefix + "partial_" + std::to_string(incompl_i) + ".voxelgrid";
        const std::string meta_filename = m_output_prefix + "partial_" + std::to_string(incompl_i) + ".metadata";
        ROS_INFO("generate_partial_environments: saving file %s", output_filename.c_str());
        if (!cleared.ToFileTernaryBinary(output_filename))
        {
          ROS_FATAL("generate_partial_environments: could not save file %s", output_filename.c_str());
          return;
        }

        Metadata out_metadata;
        out_metadata.voxel_size = metadata->voxel_size;
        out_metadata.bbox_min = metadata->bbox_min;
        out_metadata.bbox_max = metadata->bbox_max;
        if (!SaveMetadata(meta_filename, out_metadata))
        {
          ROS_FATAL("generate_partial_environments: could not save metadata file %s", meta_filename.c_str());
          return;
        }
      }

      {
        const std::string output_filename = m_output_prefix + "cropped_" + std::to_string(incompl_i) + ".voxelgrid";
        const std::string meta_filename = m_output_prefix + "cropped_" + std::to_string(incompl_i) + ".metadata";
        ROS_INFO("generate_partial_environments: saving cropped file %s", output_filename.c_str());
        Voxelgrid cropped = cleared;

        Eigen::Vector3i bbox_min_i =
            ((m_crop_bbox_min - metadata->bbox_min) / metadata->voxel_size).array().round().cast<int>();
        Eigen::Vector3i bbox_max_i =
            ((m_crop_bbox_max - metadata->bbox_min) / metadata->voxel_size).array().round().cast<int>();
        bbox_min_i = bbox_min_i.array().max(Eigen::Vector3i::Zero().array());
        bbox_max_i = bbox_max_i.array().min(cropped.GetSize().array());
        ROS_INFO_STREAM("generate_partial_environments: cropped bounding box is " << bbox_min_i.transpose()
                        << " - " << bbox_max_i.transpose());
        cropped = *cropped.GetSubmatrix(bbox_min_i, bbox_max_i - bbox_min_i + Eigen::Vector3i::Ones());

        Metadata cropped_metadata;
        cropped_metadata.voxel_size = metadata->voxel_size * 2.0f;
        cropped_metadata.bbox_max = (bbox_max_i.cast<float>() + Eigen::Vector3f::Ones() * 0.5) *
                                     metadata->voxel_size + metadata->bbox_min;
        cropped_metadata.bbox_min = (bbox_min_i.cast<float>() + Eigen::Vector3f::Ones() * 0.5) *
                                     metadata->voxel_size + metadata->bbox_min;
        if (!SaveMetadata(meta_filename, cropped_metadata))
        {
          ROS_FATAL("generate_partial_environments: could not save metadata file %s", meta_filename.c_str());
          return;
        }

        cropped = *cropped.HalveSize();
        ROS_INFO_STREAM("generate_partial_environments: cropped bounding has size " << cropped.GetSize().transpose());
        if (!cropped.ToFileTernaryBinary(output_filename))
        {
          ROS_FATAL("generate_partial_environments: could not save file %s", output_filename.c_str());
          return;
        }
      }
    }
    ROS_INFO("generate_partial_environments: done.");
  }

  private:
  ros::NodeHandle & m_nh;

  std::string m_voxelgrid_filename;
  std::string m_metadata_filename;

  std::string m_output_prefix;
  std::string m_images_prefix;

  Eigen::Vector3f m_poi_center;
  float m_poi_radius;

  Eigen::Vector3f m_crop_bbox_min;
  Eigen::Vector3f m_crop_bbox_max;

  uint64 m_select_viewpoints_min;
  uint64 m_select_viewpoints_max;

  uint64 m_num_incomplete_environments;

  ros::Timer m_timer;
};

int main(int argc, char ** argv)
{
  ros::init(argc, argv, "generate_partial_environments");

  srand(9592); // selected with fair dice roll

  ros::NodeHandle nh("~");
  GeneratePartialEnvironments gpe(nh);

  ros::spin();

  return 0;
}
