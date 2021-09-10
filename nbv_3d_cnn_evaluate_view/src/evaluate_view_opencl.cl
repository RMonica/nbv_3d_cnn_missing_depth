#ifndef __OPENCL_VERSION__
  #define global
  #define __global
  #define kernel
  #define half float

  #include "opencl_cpp_common.h"
#endif

#define SQR(x) ((x)*(x))

#ifndef NULL
  #define NULL (0)
#endif

bool Equal2(int2 a, int2 b)
{
  return a.x == b.x && a.y == b.y;
}

bool Equal3(int3 a, int3 b)
{
  return a.x == b.x && a.y == b.y && a.z == b.z;
}

bool Equal3f(float3 a, float3 b)
{
  return a.x == b.x && a.y == b.y && a.z == b.z;
}

int2 FloatToInt2(float2 f)
{
  return (int2)(f.x, f.y);
}

int2 UintToInt2(uint2 f)
{
  return (int2)(f.x, f.y);
}

float3 UintToFloat3(uint3 f)
{
  return (float3)(f.x, f.y, f.z);
}

int3 FloatToInt3(float3 f)
{
  return (int3)(f.x, f.y, f.z);
}

void SetElem3f(float3 * v, int index, float value)
{
  switch (index)
  {
    case 0: v->x = value; break;
    case 1: v->y = value; break;
    case 2: v->z = value; break;
  }
}

float GetElem3f(const float3 v, int index)
{
  switch (index)
  {
    case 0: return v.x;
    case 1: return v.y;
    case 2: return v.z;
  }

  return v.x;
}

void SetElem3i(int3 * v, int index, int value)
{
  switch (index)
  {
    case 0: v->x = value; break;
    case 1: v->y = value; break;
    case 2: v->z = value; break;
  }
}

int GetElem3i(const int3 v, int index)
{
  switch (index)
  {
    case 0: return v.x;
    case 1: return v.y;
    case 2: return v.z;
  }

  return v.x;
}

void SetElem3u(uint3 * v, int index, uint value)
{
  switch (index)
  {
    case 0: v->x = value; break;
    case 1: v->y = value; break;
    case 2: v->z = value; break;
  }
}

uint GetElem3u(const uint3 v, int index)
{
  switch (index)
  {
    case 0: return v.x;
    case 1: return v.y;
    case 2: return v.z;
  }

  return v.x;
}

float4 quat_Inverse(const float4 q)
{
  return (float4)(-q.xyz, q.w);
}

float4 quat_Mult(const float4 q1, const float4 q2)
{
  float4 q;
  q.x = (q1.w * q2.x) + (q1.x * q2.w) + (q1.y * q2.z) - (q1.z * q2.y);
  q.y = (q1.w * q2.y) - (q1.x * q2.z) + (q1.y * q2.w) + (q1.z * q2.x);
  q.z = (q1.w * q2.z) + (q1.x * q2.y) - (q1.y * q2.x) + (q1.z * q2.w);
  q.w = (q1.w * q2.w) - (q1.x * q2.x) - (q1.y * q2.y) - (q1.z * q2.z);

  return q;
}

float3 quat_ApplyRotation(const float4 q, const float3 v)
{
  const float4 ev = (float4)(v.xyz, 0.0f);
  const float4 iq = quat_Inverse(q);
  const float4 result = quat_Mult(quat_Mult(q, ev), iq);
  return result.xyz;
}

//bool CellOutEqual(const struct CellOut * a, const struct CellOut * b)
//{
//  return a->occupancy == b->occupancy &&
//         a->normal_x == b->normal_x &&
//         a->normal_y == b->normal_y &&
//         a->normal_z == b->normal_z;
//}

void kernel Raycast(global const struct CLCell * environment, const uint width, const uint height, const uint depth,
                    const uint rays_size,
                    const float4 normal_rotation_quat,
                    global const float * origins, global const float * orientations, const uint max_range,
                    global struct CLCellOut * samples)
{
  const int ray_id = get_global_id(0);

  const float3 origin = (float3)(origins[ray_id], origins[ray_id + rays_size],
                                 origins[ray_id + 2*rays_size]);

  const float3 ray_dir = (float3)(orientations[ray_id], orientations[ray_id + rays_size],
                                      orientations[ray_id + 2*rays_size]);
                                      
  struct CLCellOut env;
  env.normal_x = 0.0f;
  env.normal_y = 0.0f;
  env.normal_z = 0.0f;
  env.z = 0.0f;
  
  samples[ray_id] = env;

  uint max_range_i = max_range;
  for (uint z = 0; z < max_range_i; z++)
  {
    float3 pt = ray_dir * (float)(z) + origin;
    float3 rpt = round(pt);
    int3 ipt = FloatToInt3(rpt);

    float3 normal = (float3)(0.0f, 0.0f, 0.0f);
    
    short occupancy;

    if (ipt.x < 0 || ipt.y < 0 || ipt.z < 0 ||
        ipt.x >= width || ipt.y >= height || ipt.z >= depth)
    {
      occupancy = CLCell_OCCUPANCY_OUTSIDE;
    }
    else
    {
      const struct CLCell env1 = environment[ipt.z * width * height + ipt.y * width + ipt.x];
      occupancy = env1.occupancy;
    }

    // compute local normal
    if (occupancy == CLCell_OCCUPANCY_OCCUPIED) // -1 = certain to be occupied
    {
      #pragma unroll
      for (int coord = 0; coord < 3; coord++)
      {
        int3 ipt_next = ipt;
        SetElem3i(&ipt_next, coord, GetElem3i(ipt_next, coord) + 1);
        int3 ipt_prev = ipt;
        SetElem3i(&ipt_prev, coord, GetElem3i(ipt_prev, coord) - 1);

        if (ipt_prev.x < 0 || ipt_prev.y < 0 || ipt_prev.z < 0 ||
            ipt_prev.x >= width || ipt_prev.y >= height || ipt_prev.z >= depth)
          ipt_prev = ipt;
        if (ipt_next.x < 0 || ipt_next.y < 0 || ipt_next.z < 0 ||
            ipt_next.x >= width || ipt_next.y >= height || ipt_next.z >= depth)
          ipt_next = ipt;

        const struct CLCell env_next = environment[ipt_next.z * width * height + ipt_next.y * width + ipt_next.x];
        const struct CLCell env_prev = environment[ipt_prev.z * width * height + ipt_prev.y * width + ipt_prev.x];

        SetElem3f(&normal, coord, env_next.tsdf - env_prev.tsdf);
      }

      normal = normalize(normal);
      normal = quat_ApplyRotation(normal_rotation_quat, normal);
    }

    env.normal_x = normal.x;
    env.normal_y = normal.y;
    env.normal_z = normal.z;

    if (occupancy == CLCell_OCCUPANCY_OCCUPIED)
    {
      env.z = (float)(z);
      samples[ray_id] = env;
      break;
    }
  }
}

bool IsShortProbability(const short s)
{
  return (s >= CLCell_OCCUPANCY_UNKNOWN_0) && (s <= CLCell_OCCUPANCY_UNKNOWN_1);
}

float ProbabilityShortToFloat(const short s)
{
  return (float)(s - CLCell_OCCUPANCY_UNKNOWN_0) / (float)(CLCell_OCCUPANCY_UNKNOWN_1 - CLCell_OCCUPANCY_UNKNOWN_0);
}

//
// P(viewed | not lost at infinity AND not lost min range AND not lost observed occupied)
// P(not lost at infinity | not lost min range AND not lost observed occupied)
// P(not lost observed occupied | not lost min range)
// P(not lost min range)
//
void kernel RaycastProbabilistic(global const struct CLCell * environment,
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
                                 global float * pixel_possible_score
                                 )
{
  const int ray_id = get_global_id(0);

  const uint rays_size = image_width * image_height;

  const float3 local_ray_dir = (float3)(local_orientations[ray_id], local_orientations[ray_id + rays_size],
                                        local_orientations[ray_id + 2*rays_size]);
  const float ray_dir_z_multiplier = local_ray_dir.z;
  const float3 ray_dir = quat_ApplyRotation(orientation, local_ray_dir);

  global const struct CLMaskPixel * const mask_pixel = &(mask[ray_id]);

  float prob_reachable = 1.0f;

  float t_score = 0.0f;
  float ray_lost_infinity_prob = 1.0f;
  float ray_not_lost_min_range_prob = 1.0f;
  float ray_not_found_unknown_occupied_prob = 1.0f;

  uint max_range_i = ceil(max_range / ray_dir_z_multiplier);

  pixel_possible_score[ray_id] = (float)(max_range_i);

  for (uint r = 0; r < max_range_i; r++)
  {
    float3 pt = ray_dir * (float)(r) + origin;
    float3 rpt = round(pt);
    int3 ipt = FloatToInt3(rpt);

    const float z = (float)(r) * ray_dir_z_multiplier;

    float pw = pow(z / sensor_focal_length, 2);
    float distance_weight = min(pw, 1.0f);

    short occupancy;

    if (ipt.x < 0 || ipt.y < 0 || ipt.z < 0 ||
        ipt.x >= width || ipt.y >= height || ipt.z >= depth)
    {
      occupancy = CLCell_OCCUPANCY_OUTSIDE;
    }
    else
    {
      const struct CLCell env1 = environment[ipt.z * width * height + ipt.y * width + ipt.x];
      occupancy = env1.occupancy;
    }

    float next_prob_reachable = prob_reachable;

    const bool is_unknown = IsShortProbability(occupancy);

//    if (occupancy == CLCell_OCCUPANCY_OCCUPIED)
//    {
//      next_prob_reachable = 0.0f;
//      ray_lost_infinity_prob = 0.0f;
//      ray_not_lost_occupied_prob *= (1.0 - a_priori_ray_lost_if_occupied);
//      if (z < min_range)
//        ray_not_lost_min_range_prob = 0.0f;
//    }
    if (occupancy == CLCell_OCCUPANCY_EMPTY)
    {
      next_prob_reachable = prob_reachable;
    }

    if (is_unknown || occupancy == CLCell_OCCUPANCY_OUTSIDE)
    {
      float occ_prob;
      if (is_unknown)
        occ_prob = ProbabilityShortToFloat(occupancy);
      else
        occ_prob = a_priori_ray_occupied_if_outside;

      if (z < min_range)
        ray_not_lost_min_range_prob *= (1.0f - occ_prob);
      else
      {
        next_prob_reachable = prob_reachable * (1.0f - occ_prob);

        ray_lost_infinity_prob *= (1.0f - occ_prob); // if we hit an occupied voxel, then the ray is not lost at infinity
        ray_not_found_unknown_occupied_prob *= (1.0 - occ_prob); // it may be lost at occupied, however
      }
    }

    if (z >= mask_pixel->depth)
    {
      const ushort visibility = mask_pixel->visibility;
      if (visibility == CLMaskPixel_LOST)
      {
        break;
      }
      if (visibility == CLMaskPixel_OCCUPIED)
      {
        ray_lost_infinity_prob = 0.0f;
        if (z < min_range)
          ray_not_lost_min_range_prob = 0.0f;
        break;
      }
      if (visibility == CLMaskPixel_UNKNOWN)
      {
        // do nothing and continue
      }
    }

    const float score = prob_reachable * (is_unknown ? 1.0f : 0.0f);

    bool in_roi = true;
    if (roi_data->has_sphere_roi)
    {
      if (distance(roi_data->sphere_center, pt) > roi_data->sphere_radius)
        in_roi = false;
    }

    if (in_roi)
      t_score += score * distance_weight;

    if (next_prob_reachable == 0.0f || ray_not_lost_min_range_prob == 0.0f)
      break;
    prob_reachable = next_prob_reachable;
  }

  const float ray_lost_occupied = (1.0 - ray_not_found_unknown_occupied_prob) * a_priori_ray_lost_if_occupied;
  const float ray_not_lost_prob = (1.0f - ray_lost_infinity_prob) * ray_not_lost_min_range_prob * (1.0 - ray_lost_occupied);
  pixel_score[ray_id] = t_score * ray_not_lost_prob;
}

void kernel RaycastDumbProbabilistic(global const struct CLCell * environment,
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
                                     )
{
  const int ray_id = get_global_id(0);

  const uint rays_size = image_width * image_height;

  const float3 local_ray_dir = (float3)(local_orientations[ray_id], local_orientations[ray_id + rays_size],
                                        local_orientations[ray_id + 2*rays_size]);
  const float ray_dir_z_multiplier = local_ray_dir.z;
  const float3 ray_dir = quat_ApplyRotation(orientation, local_ray_dir);

  float prob_reachable = 1.0f;

  float t_score = 0.0f;
  float ray_not_lost_min_range_prob = 1.0f;

  uint max_range_i = ceil(max_range / ray_dir_z_multiplier);

  pixel_possible_score[ray_id] = (float)(max_range_i);

  for (uint r = 0; r < max_range_i; r++)
  {
    float3 pt = ray_dir * (float)(r) + origin;
    float3 rpt = round(pt);
    int3 ipt = FloatToInt3(rpt);

    const float z = (float)(r) * ray_dir_z_multiplier;

    float pw = pow(z / sensor_focal_length, 2);
    float distance_weight = min(pw, 1.0f);

    short occupancy;

    if (ipt.x < 0 || ipt.y < 0 || ipt.z < 0 ||
        ipt.x >= width || ipt.y >= height || ipt.z >= depth)
    {
      occupancy = CLCell_OCCUPANCY_OUTSIDE;
    }
    else
    {
      const struct CLCell env1 = environment[ipt.z * width * height + ipt.y * width + ipt.x];
      occupancy = env1.occupancy;
    }

    float next_prob_reachable = prob_reachable;

    const bool is_unknown = IsShortProbability(occupancy);

    if (occupancy == CLCell_OCCUPANCY_EMPTY)
    {
      next_prob_reachable = prob_reachable;
    }

    if (is_unknown || occupancy == CLCell_OCCUPANCY_OUTSIDE)
    {
      float occ_prob;
      if (is_unknown)
        occ_prob = ProbabilityShortToFloat(occupancy);
      else
        occ_prob = a_priori_ray_occupied_if_outside;

      if (z < min_range)
      {
        //ray_not_lost_min_range_prob *= (1.0f - occ_prob);
      }
      else
      {
        next_prob_reachable = prob_reachable * (1.0f - occ_prob);
      }
    }

    if (occupancy == CLCell_OCCUPANCY_OCCUPIED)
    {
      next_prob_reachable = 0.0f;
      if (z < min_range)
        ray_not_lost_min_range_prob = 0.0f;
    }

    const float score = prob_reachable * (is_unknown ? 1.0f : 0.0f);

    bool in_roi = true;
    if (roi_data->has_sphere_roi)
    {
      if (distance(roi_data->sphere_center, pt) > roi_data->sphere_radius)
        in_roi = false;
    }

    if (in_roi)
      t_score += score * distance_weight;

    if (next_prob_reachable == 0.0f || ray_not_lost_min_range_prob == 0.0f)
      break;
    prob_reachable = next_prob_reachable;
  }

  const float ray_not_lost_prob = ray_not_lost_min_range_prob;
  pixel_score[ray_id] = t_score * ray_not_lost_prob;
}

void kernel RaycastGTView(global const struct CLCell * environment,
                          const uint width, const uint height, const uint depth,
                          const uint image_width, const uint image_height,
                          const float4 intrinsics, // cx cy fx fy
                          const float3 origin,
                          const float4 orientation,
                          global const float * sensor_image,
                          global const struct CLROIData * roi_data,
                          global float * pixel_score
                          )
{
  const uint image_x = get_global_id(0);
  const uint image_y = get_global_id(1);
  const uint image_i = image_x + image_y * image_width;

  float cx = intrinsics.x;
  float cy = intrinsics.y;
  float fx = intrinsics.z;
  float fy = intrinsics.w;
  const float3 local_ray_dir = normalize((float3)(((float)(image_x) - cx + 0.5) / fx,
                                         ((float)(image_y) - cy + 0.5) / fy,
                                         1.0f));
  const float ray_dir_z_multiplier = local_ray_dir.z;
  const float3 ray_dir = quat_ApplyRotation(orientation, local_ray_dir);

  const float sensor_focal_length = (fx + fy) / 2.0f;

  pixel_score[image_i] = 0.0f;

  if (isnan(sensor_image[image_i]) || isinf(sensor_image[image_i]))
    return;

  const float max_sensor_depth = sensor_image[image_i];

  const uint SANITY_CHECK = max_sensor_depth / ray_dir_z_multiplier + 20;

  float t_score = 0.0f;

  for (uint r = 0; r < SANITY_CHECK; r++)
  {
    float3 pt = ray_dir * (float)(r) + origin;
    float3 rpt = round(pt);
    int3 ipt = FloatToInt3(rpt);

    const float z = (float)(r) * ray_dir_z_multiplier;

    float pw = pow(z / sensor_focal_length, 2);
    float distance_weight = min(pw, 1.0f);

    if (ipt.x < 0 || ipt.y < 0 || ipt.z < 0 ||
        ipt.x >= width || ipt.y >= height || ipt.z >= depth)
      continue;

    const struct CLCell env1 = environment[ipt.z * width * height + ipt.y * width + ipt.x];

    const bool is_unknown = IsShortProbability(env1.occupancy);

    bool in_roi = true;
    if (roi_data->has_sphere_roi)
    {
      if (distance(roi_data->sphere_center, pt) > roi_data->sphere_radius)
        in_roi = false;
    }

    if (is_unknown && in_roi)
      t_score += 1.0f * distance_weight;

    if (max_sensor_depth < z)
      break;
  }

  pixel_score[image_i] = t_score;
}

void kernel FillUint(
                     const uint c,
                     global uint * to_be_filled
                     )
{
  const int x = get_global_id(0);
  to_be_filled[x] = c;
}

void kernel FillFloat(
                      const float c,
                      global float * to_be_filled
                      )
{
  const int x = get_global_id(0);
  to_be_filled[x] = c;
}
