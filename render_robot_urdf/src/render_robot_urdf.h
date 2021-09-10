#ifndef RENDER_ROBOT_URDF_H
#define RENDER_ROBOT_URDF_H

#include <GL/glew.h>

#include <Eigen/Dense>

#include <sensor_msgs/JointState.h>
#include <ros/ros.h>

#include <memory>
#include <string>
#include <stdint.h>

namespace realtime_urdf_filter {
  class RealtimeURDFFilter;
}

class RenderRobotURDF
{
  public:
  typedef std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f> > Vector3fVector;
  typedef std::vector<float> FloatVector;
  typedef uint32_t uint32;
  typedef uint64_t uint64;
  typedef int32_t int32;
  typedef std::vector<uint32> Uint32Vector;

  typedef std::vector<GLfloat> GLfloatVector;
  typedef std::vector<GLint> GLintVector;
  typedef std::vector<GLuint> GLuintVector;


  RenderRobotURDF(ros::NodeHandle & nh,
                  const bool with_color,
                  const bool with_normals,
                  const std::string model_param_name);

  ~RenderRobotURDF();

  void Render(const Eigen::Affine3d &pose,
              const sensor_msgs::JointState * joint_states,
              const Eigen::Affine3d & robot_position);

  void SetIntrinsics(const Eigen::Vector2i & size,
                     const Eigen::Vector2f & center,
                     const Eigen::Vector2f & focal,
                     const Eigen::Vector2f & range
                     );

  Vector3fVector GetColorResult();
  Vector3fVector GetNormalResult();
  FloatVector GetDepthResult();
  Uint32Vector GetIndicesResult();

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  private:
  void CheckOpenGLError(const std::string message);
  void CheckShaderError(const GLhandleARB shader_id,const std::string message);
  void CheckLinkError(const GLint program_id, const std::string message);

  std::shared_ptr<realtime_urdf_filter::RealtimeURDFFilter> m_realtimeURDFFilter;

  ros::NodeHandle & m_nh;

  bool m_with_color;
  bool m_with_normals;

  std::string m_model_param_name;

  Eigen::Vector2i m_size;
  Eigen::Vector2f m_center;
  Eigen::Vector2f m_focal;
  Eigen::Vector2f m_range;

  GLuint m_renderbuffer_id;
  GLuint m_framebuffer_id;
  GLuint m_color_texture_id;
  GLuint m_depth_texture_id;
  GLuint m_index_texture_id;
  GLuint m_normals_texture_id;

  GLint m_cam_robot_location;
  GLint m_t_inv_robot_location;
  GLint m_maxdepth_robot_location;
  GLint m_image_size_robot_location;

  GLint m_program_robot_id;
  GLhandleARB m_vertex_shader_robot_id;
  GLhandleARB m_fragment_shader_robot_id;
};

#endif // RENDER_ROBOT_URDF_H
