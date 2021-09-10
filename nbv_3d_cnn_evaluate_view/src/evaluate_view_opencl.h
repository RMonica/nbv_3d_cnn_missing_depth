#ifndef EVALUATE_VIEW_OPENCL_H
#define EVALUATE_VIEW_OPENCL_H

// OpenCL
#include <CL/cl2.hpp>

#include <ros/ros.h>

#include <Eigen/Dense>
#include <Eigen/StdVector>

#include <memory>
#include <vector>

#include <rmonica_voxelgrid_common/voxelgrid.h>

class EvaluateViewOpenCL
{
  public:
  typedef std::shared_ptr<cl::Context> CLContextPtr;
  typedef std::shared_ptr<cl::CommandQueue> CLCommandQueuePtr;
  typedef std::shared_ptr<cl::Buffer> CLBufferPtr;
  typedef std::shared_ptr<cl::Device> CLDevicePtr;
  typedef std::shared_ptr<cl::Program> CLProgramPtr;
  typedef std::shared_ptr<cl::Kernel> CLKernelPtr;
  typedef std::vector<cl_float, Eigen::aligned_allocator<cl_float> > CLFloatVector;
  typedef std::vector<cl_float2, Eigen::aligned_allocator<cl_float2> > CLFloat2Vector;
  typedef std::vector<cl_float4, Eigen::aligned_allocator<cl_float4> > CLFloat4Vector;
  typedef std::vector<cl_int2, Eigen::aligned_allocator<cl_int2> > CLInt2Vector;
  typedef std::vector<cl_int3, Eigen::aligned_allocator<cl_int3> > CLInt3Vector;
  typedef std::vector<cl_float, Eigen::aligned_allocator<cl_uchar> > CLUCharVector;
  typedef std::vector<cl_float3, Eigen::aligned_allocator<cl_float3> > CLFloat3Vector;
  typedef std::vector<cl_int, Eigen::aligned_allocator<cl_int> > CLInt32Vector;
  typedef std::vector<cl_uint, Eigen::aligned_allocator<cl_uint> > CLUInt32Vector;
  typedef std::vector<cl_ushort, Eigen::aligned_allocator<cl_ushort> > CLUInt16Vector;
  typedef std::vector<cl_ushort2, Eigen::aligned_allocator<cl_ushort2> > CLUShort2Vector;

  typedef rmonica_voxelgrid_common::Voxelgrid Voxelgrid;

  typedef uint64_t uint64;
  typedef uint8_t uint8;
  typedef int8_t int8;
  typedef uint32_t uint32;
  typedef int32_t int32;

  typedef std::vector<Eigen::Vector2i, Eigen::aligned_allocator<Eigen::Vector2i> > Vector2iVector;
  typedef std::vector<Eigen::Vector2f, Eigen::aligned_allocator<Eigen::Vector2f> > Vector2fVector;
  typedef std::vector<Eigen::Vector3i, Eigen::aligned_allocator<Eigen::Vector3i> > Vector3iVector;
  typedef std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f> > Vector3fVector;
  typedef std::vector<Eigen::Vector4i, Eigen::aligned_allocator<Eigen::Vector4i> > Vector4iVector;
  typedef std::vector<Eigen::Quaternionf, Eigen::aligned_allocator<Eigen::Quaternionf> > QuaternionfVector;
  typedef std::vector<float> FloatVector;
  typedef std::vector<uint64> Uint64Vector;
  typedef std::vector<int8> Int8Vector;
  
  struct CellResult
  {
    float z;
    uint8 status;
    Eigen::Vector3f normal;
    
    Eigen::Vector3f local_direction;
    Eigen::Vector3f world_direction;

    enum
    {
      UNKNOWN  = 0, // the status is unknown
      OCCUPIED = 1, // the sensor sees an occupied pixel
      LOST     = 2, // the sensor sees nothing (ray is lost to infinity or below min range)
    };
  };
  
  typedef std::vector<CellResult, Eigen::aligned_allocator<CellResult> > CellResultVector;

  struct SampleRepeat
  {
    float value;
    Eigen::Vector3f normal;
    uint32 repeat;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  };

  struct RayResult
  {
    Eigen::Vector3f local_direction;
    Eigen::Vector3f world_direction;
    std::vector<SampleRepeat, Eigen::aligned_allocator<RayResult> > samples;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  };

  typedef std::vector<RayResult, Eigen::aligned_allocator<RayResult> > RayResultVector;

  struct RaycastResult
  {
    Eigen::Vector2i size;
    CellResultVector ray_results;
  };

  EvaluateViewOpenCL(ros::NodeHandle & nh);

  struct MaskPixel
  {
    float depth;
    uint32 visibility;

    enum
    {
      STATUS_LOST     = 0,
      STATUS_OCCUPIED = 1,
      STATUS_UNKNOWN  = 2,
    };
  };
  typedef std::vector<MaskPixel> MaskPixelVector;

  struct ROIData
  {
    bool has_sphere_roi;
    Eigen::Vector3f sphere_center;
    float sphere_radius;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  };

  RaycastResult Raycast(const Eigen::Vector2f & center,
                        const Eigen::Vector2f & focal,
                        const Eigen::Vector2i & size,
                        const uint64 max_range_cells,
                        const Eigen::Affine3f & pose);

  FloatVector RaycastProbabilistic(const Eigen::Vector2f & center,
                                   const Eigen::Vector2f & focal,
                                   const Eigen::Vector2i & image_size,
                                   const MaskPixelVector & mask,
                                   const float max_range_cells,
                                   const float min_range_cells,
                                   const float a_priori_ray_lost_if_occupied,
                                   const float a_priori_ray_occupied_if_outside,
                                   const ROIData & roi,
                                   const Eigen::Affine3f & pose);

  FloatVector RaycastDumbProbabilistic(const Eigen::Vector2f & center,
                                       const Eigen::Vector2f & focal,
                                       const Eigen::Vector2i & image_size,
                                       const float max_range_cells,
                                       const float min_range_cells,
                                       const float a_priori_ray_occupied_if_outside,
                                       const ROIData & roi,
                                       const Eigen::Affine3f & pose);

  FloatVector ComputeGroundTruthScores(const Eigen::Vector2f & center,
                                       const Eigen::Vector2f & focal,
                                       const Eigen::Vector2i & image_size,
                                       const FloatVector &actual_sensor_image,
                                       const ROIData & roi,
                                       const Eigen::Affine3f & pose);

  void SetEnvironment(const Eigen::Vector3i & environment_size,
                      const Int8Vector & environment,

                      const float a_priori_occupancy_probability,
                      const Eigen::Vector3i &probabilities_origin,
                      const Eigen::Vector3i &probabilities_size,
                      const FloatVector &probabilities,

                      const Vector4iVector & tsdf);

  Vector3fVector TransformRayDirs(const Vector3fVector & dirs,
                                  const Eigen::Quaternionf & orientation);

  Vector3fVector GenerateSensorRayDirs(const Eigen::Vector2f &center,
                                       const Eigen::Vector2f &sensor_f,
                                       const Eigen::Vector2i & sensor_resolution);

  private:

  void InitOpenCL(const std::string & node_name);
  CLBufferPtr CreateBuffer(const CLContextPtr context,
                           const size_t size,
                           const std::string name) const;

  ros::NodeHandle & m_nh;

  CLDevicePtr m_opencl_device;
  CLContextPtr m_opencl_context;
  CLCommandQueuePtr m_opencl_command_queue;
  CLProgramPtr m_opencl_program;

  // utilities
  CLKernelPtr m_fill_uint_kernel;

  // Environment
  CLBufferPtr m_raycast_environment;
  uint64 m_last_environment_size = 0;
  Eigen::Vector3i m_curr_environment_size;

  // Raycast
  CLKernelPtr m_raycast_kernel;
  CLBufferPtr m_raycast_origins;
  CLBufferPtr m_raycast_directions;
  uint64 m_raycast_last_origins_size = 0;
  CLBufferPtr m_raycast_samples;
  uint64 m_raycast_last_result_size = 0;

  // RaycastProbabilistic
  CLKernelPtr m_raycastp_kernel;
  CLBufferPtr m_raycastp_mask;
  CLBufferPtr m_raycastp_local_directions;
  uint64 m_raycastp_last_image_size = 0;
  CLBufferPtr m_raycastp_scores;
  CLBufferPtr m_raycastp_possible_scores;
  CLBufferPtr m_raycastp_roi_data;

  // RaycastDumbProbabilistic
  CLKernelPtr m_raycastdumbp_kernel;
  CLBufferPtr m_raycastdumbp_local_directions;
  uint64 m_raycastdumbp_last_image_size = 0;
  CLBufferPtr m_raycastdumbp_scores;
  CLBufferPtr m_raycastdumbp_possible_scores;
  CLBufferPtr m_raycastdumbp_roi_data;

  // ComputeGroundTruthScores
  CLKernelPtr m_compute_gt_kernel;
  CLBufferPtr m_compute_gt_sensor_images;
  uint64 m_compute_gt_last_image_size = 0;
  CLBufferPtr m_compute_gt_scores;
  uint64 m_compute_gt_last_scores_size = 0;
  CLBufferPtr m_compute_gt_roi_data;
};

#endif // EVALUATE_VIEW_OPENCL_H
