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

