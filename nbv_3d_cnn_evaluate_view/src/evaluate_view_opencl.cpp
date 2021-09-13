/*
 * Copyright (c) 2021, Riccardo Monica
 *   RIMLab, Department of Engineering and Architecture, University of Parma, Italy
 *   http://www.rimlab.ce.unipr.it/
 *
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted
 * provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this list of conditions
 * and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice, this list of
 * conditions and the following disclaimer in the documentation and/or other materials provided with
 * the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its contributors may be used to
 * endorse or promote products derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
 * IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
 * OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "evaluate_view_opencl.h"
#include "evaluate_view_opencl_parameters.h"

#include "evaluate_view_opencl.cl.h"

#include "opencl_cpp_common.h"

typedef std::vector<CLMaskPixel, Eigen::aligned_allocator<CLMaskPixel> > CLMaskPixelVector;

static cl_float2 EToCL(const Eigen::Vector2f & v)
{
  cl_float2 result;
  result.s[0] = v[0];
  result.s[1] = v[1];
  return result;
}

static cl_int2 EToCL(const Eigen::Vector2i & v)
{
  cl_int2 result;
  result.s[0] = v[0];
  result.s[1] = v[1];
  return result;
}

static cl_float3 EToCL(const Eigen::Vector3f & v)
{
  cl_float3 result;
  result.s[0] = v[0];
  result.s[1] = v[1];
  result.s[2] = v[2];
  return result;
}

static cl_float4 EToCL(const Eigen::Vector4f & v)
{
  cl_float4 result;
  result.s[0] = v[0];
  result.s[1] = v[1];
  result.s[2] = v[2];
  result.s[3] = v[3];
  return result;
}

static cl_int3 EToCL(const Eigen::Vector3i & v)
{
  cl_int3 result;
  result.s[0] = v[0];
  result.s[1] = v[1];
  result.s[2] = v[2];
  return result;
}

static cl_float4 EToCL(const Eigen::Quaternionf & q)
{
  cl_float4 result;
  result.s[0] = q.x();
  result.s[1] = q.y();
  result.s[2] = q.z();
  result.s[3] = q.w();
  return result;
}

static Eigen::Vector2i CLToE(const cl_int2 & v)
{
  return Eigen::Vector2i(v.s[0], v.s[1]);
}

static Eigen::Vector3i CLToE(const cl_int3 & v)
{
  return Eigen::Vector3i(v.s[0], v.s[1], v.s[2]);
}

static Eigen::Vector2f CLToE(const cl_float2 & v)
{
  return Eigen::Vector2f(v.s[0], v.s[1]);
}

static Eigen::Vector3f CLToE(const cl_float3 & v)
{
  return Eigen::Vector3f(v.s[0], v.s[1], v.s[2]);
}

EvaluateViewOpenCL::EvaluateViewOpenCL(ros::NodeHandle & nh): m_nh(nh)
{
  InitOpenCL("EvaluateViewOpenCL");
}

void EvaluateViewOpenCL::InitOpenCL(const std::string & node_name)
{
  std::string param_string;

  std::string platform_name;
  m_nh.param<std::string>(PARAM_NAME_OPENCL_PLATFORM_NAME, platform_name, PARAM_DEFAULT_OPENCL_PLATFORM_NAME);
  std::string device_name;
  m_nh.param<std::string>(PARAM_NAME_OPENCL_DEVICE_NAME, device_name, PARAM_DEFAULT_OPENCL_DEVICE_NAME);

  cl_device_type device_type;
  m_nh.param<std::string>(PARAM_NAME_OPENCL_DEVICE_TYPE, param_string, PARAM_DEFAULT_OPENCL_DEVICE_TYPE);
  if (param_string == PARAM_VALUE_OPENCL_DEVICE_TYPE_ALL)
    device_type = CL_DEVICE_TYPE_ALL;
  else if (param_string == PARAM_VALUE_OPENCL_DEVICE_TYPE_CPU)
    device_type = CL_DEVICE_TYPE_CPU;
  else if (param_string == PARAM_VALUE_OPENCL_DEVICE_TYPE_GPU)
    device_type = CL_DEVICE_TYPE_GPU;
  else
  {
    ROS_ERROR("%s: invalid parameter opencl_device_type, value '%s', using '%s' instead.",
              node_name.c_str(), param_string.c_str(), PARAM_VALUE_OPENCL_DEVICE_TYPE_ALL);
    device_type = CL_DEVICE_TYPE_ALL;
  }

  std::vector<cl::Platform> all_platforms;
  cl::Platform::get(&all_platforms);

  if (all_platforms.empty())
  {
    ROS_ERROR("%s: opencl: no platforms found.", node_name.c_str());
    exit(1);
  }

  {
    std::string all_platform_names;
    for (uint64 i = 0; i < all_platforms.size(); i++)
      all_platform_names += "\n  -- " + all_platforms[i].getInfo<CL_PLATFORM_NAME>();
    ROS_INFO_STREAM(node_name << ": opencl: found platforms:" << all_platform_names);
  }
  uint64 platform_id = 0;
  if (!platform_name.empty())
  {
    ROS_INFO("%s: looking for matching platform: %s", node_name.c_str(), platform_name.c_str());
    for (uint64 i = 0; i < all_platforms.size(); i++)
    {
      const std::string plat = all_platforms[i].getInfo<CL_PLATFORM_NAME>();
      if (plat.find(platform_name) != std::string::npos)
      {
        ROS_INFO("%s: found matching platform: %s", node_name.c_str(), plat.c_str());
        platform_id = i;
        break;
      }
    }
  }

  cl::Platform default_platform = all_platforms[platform_id];
  ROS_INFO_STREAM(node_name << ": using platform: " << default_platform.getInfo<CL_PLATFORM_NAME>());

  std::vector<cl::Device> all_devices;
  default_platform.getDevices(device_type, &all_devices);
  if (all_devices.empty())
  {
      ROS_INFO("%s: no devices found.", node_name.c_str());
      exit(1);
  }
  {
    std::string all_device_names;
    for (uint64 i = 0; i < all_devices.size(); i++)
      all_device_names += "\n  -- " + all_devices[i].getInfo<CL_DEVICE_NAME>();
    ROS_INFO_STREAM(node_name << ": found devices:" << all_device_names);
  }
  uint64 device_id = 0;
  if (!device_name.empty())
  {
    ROS_INFO("%s: looking for matching device: %s", node_name.c_str(), device_name.c_str());
    for (uint64 i = 0; i < all_devices.size(); i++)
    {
      const std::string dev = all_devices[i].getInfo<CL_DEVICE_NAME>();
      if (dev.find(device_name) != std::string::npos)
      {
        ROS_INFO("%s: found matching device: %s", node_name.c_str(), dev.c_str());
        device_id = i;
        break;
      }
    }
  }

  cl::Device default_device = all_devices[device_id];
  m_opencl_device = CLDevicePtr(new cl::Device(default_device));
  ROS_INFO_STREAM(node_name << ": using device: " << default_device.getInfo<CL_DEVICE_NAME>());

  m_opencl_context = CLContextPtr(new cl::Context({*m_opencl_device}));

  m_opencl_command_queue = CLCommandQueuePtr(new cl::CommandQueue(*m_opencl_context,*m_opencl_device));

  std::string source = EVALUATE_VIEW_OPENCL_CL;

  cl::Program::Sources sources;
  sources.push_back({source.c_str(),source.length()});

  ROS_INFO("%s: building program... ", node_name.c_str());
  m_opencl_program = CLProgramPtr(new cl::Program(*m_opencl_context,sources));
  if (m_opencl_program->build({*m_opencl_device}) != CL_SUCCESS)
  {
    ROS_ERROR_STREAM(node_name << ": error building opencl_program: " <<
                     m_opencl_program->getBuildInfo<CL_PROGRAM_BUILD_LOG>(*m_opencl_device));
    exit(1);
  }
  ROS_INFO("%s: initialized.", node_name.c_str());
}

EvaluateViewOpenCL::CLBufferPtr EvaluateViewOpenCL::CreateBuffer(const CLContextPtr context,
                                                                 const size_t size,
                                                                 const std::string name) const
{
  cl_int err;
  CLBufferPtr buf = CLBufferPtr(new cl::Buffer(*context,CL_MEM_READ_WRITE,
                                    size, NULL, &err));
  if (err != CL_SUCCESS)
  {
    ROS_ERROR("could not allocate buffer '%s' of size %u, error %d", name.c_str(), unsigned(size), int(err));
  }
  return buf;
}

/// ====================================== ///

typedef std::vector<CLCell> CLCellVector;

typedef std::vector<CLCellOut> CLCellOutVector;

void EvaluateViewOpenCL::SetEnvironment(const Eigen::Vector3i & environment_size,
                                        const Int8Vector & environment,
                                        const float a_priori_occupancy_probability,
                                        const Eigen::Vector3i & probabilities_origin,
                                        const Eigen::Vector3i & probabilities_size,
                                        const FloatVector & probabilities,
                                        const Vector4iVector & tsdf)
{
  const uint64 width = environment_size.x();
  const uint64 height = environment_size.y();
  const uint64 depth = environment_size.z();
  m_curr_environment_size = environment_size;

  if (!m_raycast_environment || environment_size.prod() != m_last_environment_size)
  {
    m_raycast_environment = CreateBuffer(m_opencl_context, environment_size.prod() * sizeof(CLCell),
                                         "m_raycast_environment");
    m_last_environment_size = environment_size.prod();
  }

  CLCellVector cl_environment(environment_size.prod());
  for (uint64 z = 0; z < depth; z++)
    for (uint64 y = 0; y < height; y++)
      for (uint64 x = 0; x < width; x++)
      {
        const uint64 i3 = z * width * height + y * width + x;
        CLCell & cell = cl_environment[i3];
        const int8 env = environment[i3];

        if (env < 0) // empty
          cell.occupancy = CLCell_OCCUPANCY_EMPTY;
        else if (env > 0)
          cell.occupancy = CLCell_OCCUPANCY_OCCUPIED;
        else
          cell.occupancy = std::round(CLCell_OCCUPANCY_UNKNOWN_0 +
                                      (CLCell_OCCUPANCY_UNKNOWN_1 - CLCell_OCCUPANCY_UNKNOWN_0) *
                                      a_priori_occupancy_probability);
        cell.tsdf = 0;
      }

  const uint64 prob_width = probabilities_size.x();
  const uint64 prob_height = probabilities_size.y();
  const uint64 prob_depth = probabilities_size.z();
  const uint64 prob_offset_x = probabilities_origin.x();
  const uint64 prob_offset_y = probabilities_origin.y();
  const uint64 prob_offset_z = probabilities_origin.z();
  for (uint64 z = 0; z < prob_depth; z++)
    for (uint64 y = 0; y < prob_height; y++)
      for (uint64 x = 0; x < prob_width; x++)
      {
        const uint64 pi3 = z * prob_width * prob_height + y * prob_width + x;
        const uint64 i3 = (z + prob_offset_z) * width * height + (y + prob_offset_y) * width + (x + prob_offset_x);
        const float prob = probabilities[pi3];

        CLCell & cell = cl_environment[i3];
        const int8 env = environment[i3];
        if (env == 0)
        {
          cell.occupancy = std::round(CLCell_OCCUPANCY_UNKNOWN_0 +
                                      (CLCell_OCCUPANCY_UNKNOWN_1 - CLCell_OCCUPANCY_UNKNOWN_0) * prob);
        }
      }

  for (const Eigen::Vector4i & pt : tsdf)
  {
    if (pt.x() >= width)
      continue;
    if (pt.y() >= height)
      continue;
    if (pt.z() >= depth)
      continue;

    const uint64 i = pt.z() * width * height + pt.y() * width + pt.x();
    cl_environment[i].tsdf = pt.w();
  }

  m_opencl_command_queue->enqueueWriteBuffer(*m_raycast_environment, CL_TRUE, 0,
                                             cl_environment.size() * sizeof(CLCell),
                                             cl_environment.data());
}

EvaluateViewOpenCL::RaycastResult EvaluateViewOpenCL::Raycast(const Eigen::Vector2f & center,
                                                              const Eigen::Vector2f & focal,
                                                              const Eigen::Vector2i & image_size,
                                                              const uint64 max_range_cells,
                                                              const Eigen::Affine3f & pose)
{
  RaycastResult result;
  result.size = image_size;

  const Eigen::Vector3f origin = pose.translation();

  const Eigen::Quaternionf orientation(pose.linear());
  const Vector3fVector local_dirs = GenerateSensorRayDirs(center, focal, image_size);
  const Vector3fVector world_dirs = TransformRayDirs(local_dirs, orientation);
  const uint64 rays_size = world_dirs.size();

  const Eigen::Vector3i environment_size = m_curr_environment_size;
  const uint64 width = environment_size.x();
  const uint64 height = environment_size.y();
  const uint64 depth = environment_size.z();

  if (!m_raycast_origins || !m_raycast_directions ||
      rays_size != m_raycast_last_origins_size)
  {
    m_raycast_origins = CreateBuffer(m_opencl_context, rays_size * 3 * sizeof(cl_float),
                                     "m_raycast_origins");
    m_raycast_directions = CreateBuffer(m_opencl_context, rays_size * 3 * sizeof(cl_float),
                                          "m_raycast_directions");
    m_raycast_last_origins_size = rays_size;
  }

  const Eigen::Vector2i raycast_result_size(image_size.x(), image_size.y());
  if (!m_raycast_samples || raycast_result_size.prod() != m_raycast_last_result_size)
  {
    m_raycast_samples = CreateBuffer(m_opencl_context, raycast_result_size.prod() * sizeof(CLCellOutVector),
                                     "m_raycast_samples");
    m_raycast_last_result_size = raycast_result_size.prod();
  }

  CLFloatVector cl_origins(rays_size * 3);
  for (uint64 i = 0; i < rays_size; i++)
  {
    cl_origins[i + 0 * rays_size] = origin.x();
    cl_origins[i + 1 * rays_size] = origin.y();
    cl_origins[i + 2 * rays_size] = origin.z();
  }

  m_opencl_command_queue->enqueueWriteBuffer(*m_raycast_origins, CL_TRUE, 0,
                                             cl_origins.size() * sizeof(cl_float),
                                             cl_origins.data());

  CLFloatVector cl_dirs(rays_size * 3);
  for (uint64 i = 0; i < rays_size; i++)
  {
    cl_dirs[i + 0 * rays_size] = world_dirs[i].x();
    cl_dirs[i + 1 * rays_size] = world_dirs[i].y();
    cl_dirs[i + 2 * rays_size] = world_dirs[i].z();
  }

  m_opencl_command_queue->enqueueWriteBuffer(*m_raycast_directions, CL_TRUE, 0,
                                             cl_dirs.size() * sizeof(cl_float),
                                             cl_dirs.data());

  /*
   * global const float * environment, const uint width, const uint height, const uint depth,
                    const uint rays_size,
                    const float4 normal_rotation_quat,
                    global const float * origins, global const float * orientations, const float max_range,
                    global float * samples, global uint * samples_repeat, global int * samples_counters*/
  if (!m_raycast_kernel)
  {
    m_raycast_kernel = CLKernelPtr(new cl::Kernel(*m_opencl_program, "Raycast"));
  }

  const Eigen::Quaternionf normal_rotation_quat(pose.linear().transpose());
  const cl_float4 cl_normal_rotation_quat{normal_rotation_quat.x(), normal_rotation_quat.y(),
                                          normal_rotation_quat.z(), normal_rotation_quat.w()};

  {
    uint64 c = 0;
    m_raycast_kernel->setArg(c++, *m_raycast_environment);
    m_raycast_kernel->setArg(c++, cl_uint(width));
    m_raycast_kernel->setArg(c++, cl_uint(height));
    m_raycast_kernel->setArg(c++, cl_uint(depth));
    m_raycast_kernel->setArg(c++, cl_uint(rays_size));
    m_raycast_kernel->setArg(c++, cl_float4(cl_normal_rotation_quat));
    m_raycast_kernel->setArg(c++, *m_raycast_origins);
    m_raycast_kernel->setArg(c++, *m_raycast_directions);
    m_raycast_kernel->setArg(c++, cl_uint(max_range_cells));
    m_raycast_kernel->setArg(c++, *m_raycast_samples);
  }

  int ret = m_opencl_command_queue->enqueueNDRangeKernel(*m_raycast_kernel, cl::NullRange,
                                    cl::NDRange(rays_size), cl::NullRange);
  if (ret != CL_SUCCESS)
  {
    ROS_ERROR("  EvaluateViewOpenCL::Raycast: error m_raycast_kernel: %d!", ret);
    exit(1);
  }

  CLCellOutVector cl_raycast_samples(raycast_result_size.prod());
  m_opencl_command_queue->enqueueReadBuffer(*m_raycast_samples, CL_TRUE, 0,
                                            cl_raycast_samples.size() * sizeof(CLCellOut),
                                            cl_raycast_samples.data());

  result.ray_results.resize(raycast_result_size.y() * raycast_result_size.x());
  for (uint64 y = 0; y < raycast_result_size.y(); y++)
    for (uint64 x = 0; x < raycast_result_size.x(); x++)
    {
      const uint64 i2 = x + y * raycast_result_size.x();
      CellResult & ray = result.ray_results[i2];
      ray.world_direction = world_dirs[i2];
      ray.local_direction = local_dirs[i2];
      CLCellOut & cell = cl_raycast_samples[i2];
      
      ray.z = cell.z;
      ray.status = cell.z ? CellResult::OCCUPIED : CellResult::UNKNOWN;
      ray.normal = Eigen::Vector3f(cell.normal_x, cell.normal_y, cell.normal_z);
    }

  return result;
}

EvaluateViewOpenCL::FloatVector EvaluateViewOpenCL::RaycastDumbProbabilistic(const Eigen::Vector2f & center,
                                                                             const Eigen::Vector2f & focal,
                                                                             const Eigen::Vector2i & image_size,
                                                                             const float max_range_cells,
                                                                             const float min_range_cells,
                                                                             const float a_priori_ray_occupied_if_outside,
                                                                             const ROIData & roi,
                                                                             const Eigen::Affine3f & pose)
{
  const uint64 rays_size = image_size.prod();
  FloatVector result(rays_size, 0.0f);

  const Eigen::Vector3f origin = pose.translation();

  const Eigen::Quaternionf orientation(pose.linear());
  const Vector3fVector local_dirs = GenerateSensorRayDirs(center, focal, image_size);
  //const Vector3fVector world_dirs = TransformRayDirs(local_dirs, orientation);

  const Eigen::Vector3i environment_size = m_curr_environment_size;
  const uint64 width = environment_size.x();
  const uint64 height = environment_size.y();
  const uint64 depth = environment_size.z();

  if (!m_raycastdumbp_local_directions ||
      !m_raycastdumbp_possible_scores || !m_raycastdumbp_scores ||
      rays_size != m_raycastdumbp_last_image_size)
  {
    m_raycastdumbp_local_directions = CreateBuffer(m_opencl_context, rays_size * 3 * sizeof(cl_float),
                                               "m_raycastdumbp_local_directions");
    m_raycastdumbp_possible_scores = CreateBuffer(m_opencl_context, rays_size * sizeof(cl_float),
                                             "m_raycastdumbp_possible_scores");
    m_raycastdumbp_scores = CreateBuffer(m_opencl_context, rays_size * sizeof(cl_float),
                                     "m_raycastdumbp_scores");
    m_raycastdumbp_last_image_size = rays_size;
  }

  if (!m_raycastdumbp_roi_data)
  {
    m_raycastdumbp_roi_data = CreateBuffer(m_opencl_context, sizeof(CLROIData),
                                       "m_raycastdumbp_roi_data");
  }

  CLFloatVector cl_local_dirs(rays_size * 3);
  for (uint64 i = 0; i < rays_size; i++)
  {
    cl_local_dirs[i + 0 * rays_size] = local_dirs[i].x();
    cl_local_dirs[i + 1 * rays_size] = local_dirs[i].y();
    cl_local_dirs[i + 2 * rays_size] = local_dirs[i].z();
  }

  m_opencl_command_queue->enqueueWriteBuffer(*m_raycastdumbp_local_directions, CL_TRUE, 0,
                                             cl_local_dirs.size() * sizeof(cl_float),
                                             cl_local_dirs.data());

  CLROIData cl_roi_data;
  cl_roi_data.has_sphere_roi = roi.has_sphere_roi;
  cl_roi_data.sphere_center = EToCL(roi.sphere_center);
  cl_roi_data.sphere_radius = roi.sphere_radius;
  m_opencl_command_queue->enqueueWriteBuffer(*m_raycastdumbp_roi_data, CL_TRUE, 0,
                                             sizeof(CLROIData),
                                             &cl_roi_data);

  /*
   *                                 global const struct CLCell * environment,
                                     const uint width, const uint height, const uint depth,
                                     const uint image_width, const uint image_height,
                                     const float sensor_focal_length,
                                     const float3 origin,
                                     global const float * local_orientations,
                                     const float4 orientation,
                                     const float min_range,
                                     const float max_range,
                                     const float a_priori_ray_occupied_if_outside,
                                     global const struct CLROIData * roi_data,
                                     global float * pixel_score,
                                     global float * pixel_possible_score
                                      */
  if (!m_raycastdumbp_kernel)
  {
    m_raycastdumbp_kernel = CLKernelPtr(new cl::Kernel(*m_opencl_program, "RaycastDumbProbabilistic"));
  }

  {
    uint64 c = 0;
    m_raycastdumbp_kernel->setArg(c++, *m_raycast_environment);
    m_raycastdumbp_kernel->setArg(c++, cl_uint(width));
    m_raycastdumbp_kernel->setArg(c++, cl_uint(height));
    m_raycastdumbp_kernel->setArg(c++, cl_uint(depth));
    m_raycastdumbp_kernel->setArg(c++, cl_uint(image_size.x()));
    m_raycastdumbp_kernel->setArg(c++, cl_uint(image_size.y()));
    m_raycastdumbp_kernel->setArg(c++, cl_float((focal.x() + focal.y()) / 2.0f));
    m_raycastdumbp_kernel->setArg(c++, cl_float3(EToCL(origin)));
    m_raycastdumbp_kernel->setArg(c++, *m_raycastdumbp_local_directions);
    m_raycastdumbp_kernel->setArg(c++, cl_float4(EToCL(orientation)));
    m_raycastdumbp_kernel->setArg(c++, cl_float(min_range_cells));
    m_raycastdumbp_kernel->setArg(c++, cl_float(max_range_cells));
    m_raycastdumbp_kernel->setArg(c++, cl_float(a_priori_ray_occupied_if_outside));
    m_raycastdumbp_kernel->setArg(c++, *m_raycastdumbp_roi_data);
    m_raycastdumbp_kernel->setArg(c++, *m_raycastdumbp_scores);
    m_raycastdumbp_kernel->setArg(c++, *m_raycastdumbp_possible_scores);
  }

  int ret = m_opencl_command_queue->enqueueNDRangeKernel(*m_raycastdumbp_kernel, cl::NullRange,
                                    cl::NDRange(rays_size), cl::NullRange);
  if (ret != CL_SUCCESS)
  {
    ROS_ERROR("  EvaluateViewOpenCL::Raycast: error m_raycastp_kernel: %d!", ret);
    exit(1);
  }

  CLFloatVector cl_scores(rays_size);
  m_opencl_command_queue->enqueueReadBuffer(*m_raycastdumbp_scores, CL_TRUE, 0,
                                            cl_scores.size() * sizeof(cl_float),
                                            cl_scores.data());

  result.resize(rays_size);
  for (uint64 i = 0; i < rays_size; i++)
    result[i] = cl_scores[i];

  return result;
}

EvaluateViewOpenCL::FloatVector EvaluateViewOpenCL::RaycastProbabilistic(const Eigen::Vector2f & center,
                                                                         const Eigen::Vector2f & focal,
                                                                         const Eigen::Vector2i & image_size,
                                                                         const MaskPixelVector & mask,
                                                                         const float max_range_cells,
                                                                         const float min_range_cells,
                                                                         const float a_priori_ray_lost_if_occupied,
                                                                         const float a_priori_ray_occupied_if_outside,
                                                                         const ROIData & roi,
                                                                         const Eigen::Affine3f & pose)
{
  const uint64 rays_size = image_size.prod();
  FloatVector result(rays_size, 0.0f);

  const Eigen::Vector3f origin = pose.translation();

  const Eigen::Quaternionf orientation(pose.linear());
  const Vector3fVector local_dirs = GenerateSensorRayDirs(center, focal, image_size);
  //const Vector3fVector world_dirs = TransformRayDirs(local_dirs, orientation);

  const Eigen::Vector3i environment_size = m_curr_environment_size;
  const uint64 width = environment_size.x();
  const uint64 height = environment_size.y();
  const uint64 depth = environment_size.z();

  if (!m_raycastp_mask || !m_raycastp_local_directions ||
      !m_raycastp_possible_scores || !m_raycastp_scores ||
      rays_size != m_raycastp_last_image_size)
  {
    m_raycastp_mask = CreateBuffer(m_opencl_context, rays_size * sizeof(CLMaskPixel),
                                     "m_raycastp_mask");
    m_raycastp_local_directions = CreateBuffer(m_opencl_context, rays_size * 3 * sizeof(cl_float),
                                               "m_raycastp_local_directions");
    m_raycastp_possible_scores = CreateBuffer(m_opencl_context, rays_size * sizeof(cl_float),
                                             "m_raycastp_possible_scores");
    m_raycastp_scores = CreateBuffer(m_opencl_context, rays_size * sizeof(cl_float),
                                     "m_raycastp_scores");
    m_raycastp_last_image_size = rays_size;
  }

  if (!m_raycastp_roi_data)
  {
    m_raycastp_roi_data = CreateBuffer(m_opencl_context, sizeof(CLROIData),
                                       "m_raycastp_roi_data");
  }

  CLFloatVector cl_local_dirs(rays_size * 3);
  for (uint64 i = 0; i < rays_size; i++)
  {
    cl_local_dirs[i + 0 * rays_size] = local_dirs[i].x();
    cl_local_dirs[i + 1 * rays_size] = local_dirs[i].y();
    cl_local_dirs[i + 2 * rays_size] = local_dirs[i].z();
  }

  m_opencl_command_queue->enqueueWriteBuffer(*m_raycastp_local_directions, CL_TRUE, 0,
                                             cl_local_dirs.size() * sizeof(cl_float),
                                             cl_local_dirs.data());

  CLROIData cl_roi_data;
  cl_roi_data.has_sphere_roi = roi.has_sphere_roi;
  cl_roi_data.sphere_center = EToCL(roi.sphere_center);
  cl_roi_data.sphere_radius = roi.sphere_radius;
  m_opencl_command_queue->enqueueWriteBuffer(*m_raycastp_roi_data, CL_TRUE, 0,
                                             sizeof(CLROIData),
                                             &cl_roi_data);

  CLMaskPixelVector cl_mask(rays_size);
  for (uint64 i = 0; i < rays_size; i++)
  {
    CLMaskPixel & cl_pix = cl_mask[i];
    const MaskPixel & pix = mask[i];

    cl_pix.depth = pix.depth;
    cl_pix.visibility = pix.visibility;
  }
  m_opencl_command_queue->enqueueWriteBuffer(*m_raycastp_mask, CL_TRUE, 0,
                                             cl_mask.size() * sizeof(CLMaskPixel),
                                             cl_mask.data());

  /*
   *                             global const struct CLCell * environment,
                                 const uint width, const uint height, const uint depth,
                                 const uint image_width, const uint image_height,
                                 const float sensor_focal_length,
                                 global const struct CLMaskPixel * mask,
                                 const float3 origin,
                                 global const float * local_orientations,
                                 const float4 orientation,
                                 const float min_range,
                                 const float max_range,
                                 const float a_priori_ray_lost_if_occupied,
                                 const float a_priori_ray_occupied_if_outside,
                                 global const struct CLROIData * roi_data,
                                 global float * pixel_score,
                                 global float * pixel_possible_score */
  if (!m_raycastp_kernel)
  {
    m_raycastp_kernel = CLKernelPtr(new cl::Kernel(*m_opencl_program, "RaycastProbabilistic"));
  }

  {
    uint64 c = 0;
    m_raycastp_kernel->setArg(c++, *m_raycast_environment);
    m_raycastp_kernel->setArg(c++, cl_uint(width));
    m_raycastp_kernel->setArg(c++, cl_uint(height));
    m_raycastp_kernel->setArg(c++, cl_uint(depth));
    m_raycastp_kernel->setArg(c++, cl_uint(image_size.x()));
    m_raycastp_kernel->setArg(c++, cl_uint(image_size.y()));
    m_raycastp_kernel->setArg(c++, cl_float((focal.x() + focal.y()) / 2.0f));
    m_raycastp_kernel->setArg(c++, *m_raycastp_mask);
    m_raycastp_kernel->setArg(c++, cl_float3(EToCL(origin)));
    m_raycastp_kernel->setArg(c++, *m_raycastp_local_directions);
    m_raycastp_kernel->setArg(c++, cl_float4(EToCL(orientation)));
    m_raycastp_kernel->setArg(c++, cl_float(min_range_cells));
    m_raycastp_kernel->setArg(c++, cl_float(max_range_cells));
    m_raycastp_kernel->setArg(c++, cl_float(a_priori_ray_lost_if_occupied));
    m_raycastp_kernel->setArg(c++, cl_float(a_priori_ray_occupied_if_outside));
    m_raycastp_kernel->setArg(c++, *m_raycastp_roi_data);
    m_raycastp_kernel->setArg(c++, *m_raycastp_scores);
    m_raycastp_kernel->setArg(c++, *m_raycastp_possible_scores);
  }

  int ret = m_opencl_command_queue->enqueueNDRangeKernel(*m_raycastp_kernel, cl::NullRange,
                                    cl::NDRange(rays_size), cl::NullRange);
  if (ret != CL_SUCCESS)
  {
    ROS_ERROR("  EvaluateViewOpenCL::Raycast: error m_raycastp_kernel: %d!", ret);
    exit(1);
  }

  CLFloatVector cl_scores(rays_size);
  m_opencl_command_queue->enqueueReadBuffer(*m_raycastp_scores, CL_TRUE, 0,
                                            cl_scores.size() * sizeof(cl_float),
                                            cl_scores.data());

  result.resize(rays_size);
  for (uint64 i = 0; i < rays_size; i++)
    result[i] = cl_scores[i];

  return result;
}

EvaluateViewOpenCL::FloatVector EvaluateViewOpenCL::ComputeGroundTruthScores(const Eigen::Vector2f & center,
                                                                             const Eigen::Vector2f & focal,
                                                                             const Eigen::Vector2i & image_size,
                                                                             const FloatVector & actual_sensor_image,
                                                                             const ROIData & roi,
                                                                             const Eigen::Affine3f & pose)
{
  const uint64 image_size_prod = image_size.prod();

  FloatVector result(image_size.prod(), 0.0f);

  const Eigen::Vector3f origin = pose.translation();

  const Eigen::Quaternionf orientation(pose.linear());

  const Eigen::Vector3i environment_size = m_curr_environment_size;
  const uint64 width = environment_size.x();
  const uint64 height = environment_size.y();
  const uint64 depth = environment_size.z();

  if (!m_compute_gt_sensor_images || m_compute_gt_last_image_size != image_size_prod)
  {
    m_compute_gt_sensor_images = CreateBuffer(m_opencl_context, image_size_prod * sizeof(cl_float),
                                              "m_raycastp_local_directions");
    m_compute_gt_last_image_size = image_size_prod;
  }

  if (!m_compute_gt_scores || m_compute_gt_last_scores_size != image_size_prod)
  {
    m_compute_gt_scores = CreateBuffer(m_opencl_context, image_size.prod() * sizeof(cl_float),
                                       "m_compute_gt_scores");
    m_compute_gt_last_scores_size = image_size_prod;
  }

  if (!m_compute_gt_roi_data)
  {
    m_compute_gt_roi_data = CreateBuffer(m_opencl_context, sizeof(CLROIData),
                                         "m_compute_gt_roi_data");
  }

  {
    CLFloatVector cl_img(image_size_prod);
    for (uint64 h = 0; h < image_size_prod; h++)
      cl_img[h] = actual_sensor_image[h];
    m_opencl_command_queue->enqueueWriteBuffer(*m_compute_gt_sensor_images, CL_TRUE, 0,
                                               cl_img.size() * sizeof(cl_float),
                                               cl_img.data());
  }

  CLROIData cl_roi_data;
  cl_roi_data.has_sphere_roi = roi.has_sphere_roi;
  cl_roi_data.sphere_center = EToCL(roi.sphere_center);
  cl_roi_data.sphere_radius = roi.sphere_radius;
  m_opencl_command_queue->enqueueWriteBuffer(*m_compute_gt_roi_data, CL_TRUE, 0,
                                             sizeof(CLROIData),
                                             &cl_roi_data);

  const Eigen::Vector4f intrinsics(center.x(), center.y(), focal.x(), focal.y());

  /*
      void kernel RaycastGTView(global const struct CLCell * environment,
                                const uint width, const uint height, const uint depth,
                                const uint image_width, const uint image_height,
                                const float4 intrinsics, // cx cy fx fy
                                const float3 origin,
                                const float4 orientation,
                                global const float * sensor_image,
                                global const struct CLROIData * roi_data,
                                global float * pixel_score
                                )*/
  if (!m_compute_gt_kernel)
  {
    m_compute_gt_kernel = CLKernelPtr(new cl::Kernel(*m_opencl_program, "RaycastGTView"));
  }

  {
    uint64 c = 0;
    m_compute_gt_kernel->setArg(c++, *m_raycast_environment);
    m_compute_gt_kernel->setArg(c++, cl_uint(width));
    m_compute_gt_kernel->setArg(c++, cl_uint(height));
    m_compute_gt_kernel->setArg(c++, cl_uint(depth));
    m_compute_gt_kernel->setArg(c++, cl_uint(image_size.x()));
    m_compute_gt_kernel->setArg(c++, cl_uint(image_size.y()));
    m_compute_gt_kernel->setArg(c++, cl_float4(EToCL(intrinsics)));
    m_compute_gt_kernel->setArg(c++, cl_float3(EToCL(origin)));
    m_compute_gt_kernel->setArg(c++, cl_float4(EToCL(orientation)));
    m_compute_gt_kernel->setArg(c++, *m_compute_gt_sensor_images);
    m_compute_gt_kernel->setArg(c++, *m_compute_gt_roi_data);
    m_compute_gt_kernel->setArg(c++, *m_compute_gt_scores);
  }

  int ret = m_opencl_command_queue->enqueueNDRangeKernel(*m_compute_gt_kernel, cl::NullRange,
                                    cl::NDRange(image_size.x(), image_size.y()), cl::NullRange);
  if (ret != CL_SUCCESS)
  {
    ROS_ERROR("  EvaluateViewOpenCL::ComputeGroundTruthScores: error m_compute_gt_kernel: %d!", ret);
    exit(1);
  }

  CLFloatVector cl_scores(image_size_prod);
  m_opencl_command_queue->enqueueReadBuffer(*m_compute_gt_scores, CL_TRUE, 0,
                                            cl_scores.size() * sizeof(cl_float),
                                            cl_scores.data());

  result.resize(image_size_prod);
  for (uint64 i = 0; i < image_size_prod; i++)
    result[i] = cl_scores[i];

  return result;
}

EvaluateViewOpenCL::Vector3fVector EvaluateViewOpenCL::TransformRayDirs(const Vector3fVector &dirs,
                                                                        const Eigen::Quaternionf & orientation)
{
  Vector3fVector result(dirs.size());
  for (uint64 i = 0; i < dirs.size(); i++)
    result[i] = orientation * dirs[i];
  return result;
}

EvaluateViewOpenCL::Vector3fVector EvaluateViewOpenCL::GenerateSensorRayDirs(
  const Eigen::Vector2f & center,
  const Eigen::Vector2f & sensor_f, const Eigen::Vector2i & sensor_resolution)
{
  Vector3fVector ray_dir(sensor_resolution.prod());

  for (uint64 iy = 0; iy < sensor_resolution.y(); iy++)
    for (uint64 ix = 0; ix < sensor_resolution.x(); ix++)
    {
      const Eigen::Vector3f v = Eigen::Vector3f((float(ix) - center.x() + 0.5f) / sensor_f.x(),
                                                (float(iy) - center.y() + 0.5f) / sensor_f.y(),
                                                1.0f
                                                ).normalized();
      ray_dir[ix + iy * sensor_resolution.x()] = v;
    }

  return ray_dir;
}

