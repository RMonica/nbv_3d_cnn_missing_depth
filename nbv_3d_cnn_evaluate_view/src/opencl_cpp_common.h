#ifdef __OPENCL_VERSION__
  typedef short cl_short;
  typedef float cl_float;
  typedef float2 cl_float2;
  typedef float3 cl_float3;
  typedef uchar cl_uchar;
  typedef uint cl_uint;
#endif
#ifndef __OPENCL_VERSION__
#include <CL/cl2.hpp>
#endif

struct __attribute__ ((packed)) CLCell
{
  cl_short tsdf;
  #define CLCell_OCCUPANCY_OUTSIDE     (-2)
  #define CLCell_OCCUPANCY_OCCUPIED    (-1)
  #define CLCell_OCCUPANCY_EMPTY       (0)
  #define CLCell_OCCUPANCY_UNKNOWN_0   (1)
  #define CLCell_OCCUPANCY_UNKNOWN_1   (10001)
  cl_short occupancy;
};

struct __attribute__ ((packed)) CLCellOut
{
  cl_float normal_x, normal_y, normal_z;
  cl_float z;
};

struct __attribute__ ((packed)) CLMaskPixel
{
  cl_float depth;
  #define CLMaskPixel_LOST     (0)
  #define CLMaskPixel_OCCUPIED (1)
  #define CLMaskPixel_UNKNOWN  (2)
  cl_uint visibility;
};

struct __attribute__ ((packed)) CLROIData
{
  cl_uchar has_sphere_roi;
  cl_float sphere_radius;
  cl_float3 sphere_center;
};

