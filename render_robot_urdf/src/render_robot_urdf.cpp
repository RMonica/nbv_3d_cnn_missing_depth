#include "render_robot_urdf.h"

#include "shaders/surfel_vertex_shader_robot.h"
#include "shaders/surfel_fragment_shader_robot.h"

//Realtime URDF Filter
#include "urdf_filter.h"
#include "urdf_renderer.h"

RenderRobotURDF::RenderRobotURDF(ros::NodeHandle & nh,
                                 const bool with_color,
                                 const bool with_normals,
                                 const std::string model_param_name): m_nh(nh)
{
  m_program_robot_id = 0;
  m_vertex_shader_robot_id = 0;
  m_fragment_shader_robot_id = 0;

  m_with_color = with_color;
  m_with_normals = with_normals;

  m_model_param_name = model_param_name;
}

RenderRobotURDF::~RenderRobotURDF()
{
  if (m_fragment_shader_robot_id)
    glDeleteShader(m_fragment_shader_robot_id);
  if (m_vertex_shader_robot_id)
    glDeleteShader(m_vertex_shader_robot_id);
  if (m_program_robot_id)
    glDeleteProgram(m_program_robot_id);
}

void RenderRobotURDF::Render(const Eigen::Affine3d & pose,
                             const sensor_msgs::JointState * joint_states,
                             const Eigen::Affine3d & robot_position)
{
  if (!joint_states)
    return;

  const Eigen::Affine3f pose_inv = pose.cast<float>().inverse();
  const Eigen::Matrix4f t_inv = pose_inv.matrix();
  const Eigen::Vector2f range = m_range;
  const Eigen::Vector2f center = m_center;
  const Eigen::Vector2f focal = m_focal;
  const Eigen::Vector4f cam(center.x(), center.y(), focal.x(), focal.y());

  glEnable(GL_DEPTH_TEST);
  glDepthFunc(GL_LESS);

  {
    GLuintVector empty_indices(m_size.x() * m_size.y(), 0);
    glBindTexture(GL_TEXTURE_2D, m_index_texture_id);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, m_size.x(), m_size.y(), GL_RED_INTEGER, GL_UNSIGNED_INT,
                    (void *)empty_indices.data());
  }

  glBindFramebuffer(GL_FRAMEBUFFER_EXT, m_framebuffer_id);

  glClearColor(0.0, 0.0, 0.0, 0.0);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  glPushAttrib(GL_VIEWPORT_BIT);

  glViewport(0, 0, m_size.x(), m_size.y());

  glUseProgram(m_program_robot_id);

  glUniform1f(m_maxdepth_robot_location, GLfloat(range.y()));
  glUniformMatrix4fv(m_t_inv_robot_location, 1, GL_FALSE, t_inv.data());
  glUniform4fv(m_cam_robot_location, 1, cam.data());
  glUniform2i(m_image_size_robot_location, GLint(m_size.x()), GLint(m_size.y()));
  CheckOpenGLError("set uniform in robot program");
  m_realtimeURDFFilter->renderRobot(*joint_states,robot_position);
  CheckOpenGLError("render robot");

  glFinish();

  glUseProgram(0);

  glPopAttrib();

  CheckOpenGLError("render");
}

RenderRobotURDF::Vector3fVector RenderRobotURDF::GetColorResult()
{
  GLfloatVector colors;

  colors.resize(m_size.x() * m_size.y() * 4);
  glBindTexture(GL_TEXTURE_2D, m_color_texture_id);
  glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT, (GLvoid *)(colors.data()));
  glBindTexture(GL_TEXTURE_2D, 0);

  CheckOpenGLError("download color texture");

  Vector3fVector result(m_size.x() * m_size.y());
  for (int32 y = 0; y < m_size.y(); y++)
    for (int32 x = 0; x < m_size.x(); x++)
    {
      const float v0 = colors[(x + y * m_size.x()) * 4 + 0];
      const float v1 = colors[(x + y * m_size.x()) * 4 + 1];
      const float v2 = colors[(x + y * m_size.x()) * 4 + 2];

      const Eigen::Vector3f c(v0, v1, v2);
      result[x + y * m_size.x()] = c;
    }

  return result;
}

RenderRobotURDF::Vector3fVector RenderRobotURDF::GetNormalResult()
{
  GLfloatVector normals;

  normals.resize(m_size.x() * m_size.y() * 3);
  glBindTexture(GL_TEXTURE_2D, m_normals_texture_id);
  glGetTexImage(GL_TEXTURE_2D, 0, GL_RGB, GL_FLOAT, (GLvoid *)(normals.data()));
  glBindTexture(GL_TEXTURE_2D, 0);

  CheckOpenGLError("download normals texture");

  Vector3fVector result(m_size.x() * m_size.y());
  for (int32 y = 0; y < m_size.y(); y++)
    for (int32 x = 0; x < m_size.x(); x++)
    {
      const float v0 = normals[(x + y * m_size.x()) * 3 + 0];
      const float v1 = normals[(x + y * m_size.x()) * 3 + 1];
      const float v2 = normals[(x + y * m_size.x()) * 3 + 2];

      const Eigen::Vector3f n(v0, v1, v2);
      result[x + y * m_size.x()] = n;
    }

  return result;
}

RenderRobotURDF::FloatVector RenderRobotURDF::GetDepthResult()
{
  FloatVector result;
  GLfloatVector depths;

  depths.resize(m_size.x() * m_size.y());
  glBindTexture(GL_TEXTURE_2D, m_depth_texture_id);
  glGetTexImage(GL_TEXTURE_2D, 0, GL_RED, GL_FLOAT, (GLvoid *)(depths.data()));
  glBindTexture(GL_TEXTURE_2D, 0);

  CheckOpenGLError("download depth texture");

  result.resize(m_size.x() * m_size.y());
  for (uint64 i = 0; i < result.size(); i++)
    result[i] = depths[i];

  return result;
}

RenderRobotURDF::Uint32Vector RenderRobotURDF::GetIndicesResult()
{
  Uint32Vector result;
  GLuintVector indices;

  indices.resize(m_size.x() * m_size.y());
  glBindTexture(GL_TEXTURE_2D, m_index_texture_id);
  glGetTexImage(GL_TEXTURE_2D, 0, GL_RED_INTEGER, GL_UNSIGNED_INT, (GLvoid *)(indices.data()));
  glBindTexture(GL_TEXTURE_2D, 0);

  CheckOpenGLError("download indices texture");

  result.resize(m_size.x() * m_size.y());
  for (uint64 i = 0; i < result.size(); i++)
    result[i] = indices[i];

  return result;
}

void RenderRobotURDF::SetIntrinsics(const Eigen::Vector2i & size,
                                    const Eigen::Vector2f & center,
                                    const Eigen::Vector2f & focal,
                                    const Eigen::Vector2f & range
                                    )
{
  m_center = center;
  m_focal = focal;
  m_range = range;

  if (size == m_size)
    return;

  m_size = size;

  if (m_fragment_shader_robot_id)
    glDeleteShader(m_fragment_shader_robot_id);
  if (m_vertex_shader_robot_id)
    glDeleteShader(m_vertex_shader_robot_id);
  if (m_program_robot_id)
    glDeleteProgram(m_program_robot_id);

  m_realtimeURDFFilter.reset(new realtime_urdf_filter::RealtimeURDFFilter(m_nh, m_size, m_model_param_name));

  // VIRTUAL VIEWPORT
  // generate frame buffer
  glGenFramebuffers(1, &m_framebuffer_id);
  CheckOpenGLError("glGenFramebuffers");

  //std::cout << "opengl ok" << std::endl;

  // generate render buffer
  glGenRenderbuffersEXT(1, &m_renderbuffer_id);
  glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, m_renderbuffer_id);
  glRenderbufferStorageEXT(GL_RENDERBUFFER_EXT, GL_DEPTH_COMPONENT24, m_size.x(), m_size.y());
  glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, 0);
  CheckOpenGLError("glGenRenderbuffers");
  //std::cout << "opengl ok" << std::endl;

  // attach render buffer
  glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, m_framebuffer_id);
  glFramebufferRenderbufferEXT(GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT, GL_RENDERBUFFER_EXT, m_renderbuffer_id);
  glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
  CheckOpenGLError("attach render buffer");
  //std::cout << "opengl ok" << std::endl;

  // generate color texture
  glGenTextures(1, &m_color_texture_id);
  CheckOpenGLError("generate color texture");
  //std::cout << "opengl ok" << std::endl;

  glBindTexture(GL_TEXTURE_2D, m_color_texture_id);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, m_size.x(), m_size.y(), 0, GL_LUMINANCE, GL_FLOAT, NULL);

  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glBindTexture(GL_TEXTURE_2D, 0);
  CheckOpenGLError("configure color texture");
  //std::cout << "opengl ok" << std::endl;

  // generate depth texture
  m_depth_texture_id = 0;
  glGenTextures(1, &m_depth_texture_id);
  CheckOpenGLError("generate depth texture");
  //std::cout << "opengl ok" << std::endl;

  glBindTexture(GL_TEXTURE_2D, m_depth_texture_id);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, m_size.x(), m_size.y(), 0, GL_LUMINANCE, GL_FLOAT, NULL);

  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glBindTexture(GL_TEXTURE_2D, 0);
  CheckOpenGLError("configure depth texture");
  //std::cout << "opengl ok" << std::endl;

  glGenTextures(1, &m_index_texture_id);
  CheckOpenGLError("generate index texture");
  //std::cout << "opengl ok" << std::endl;

  glBindTexture(GL_TEXTURE_2D, m_index_texture_id);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_R32UI, m_size.x(), m_size.y(), 0, GL_RED_INTEGER, GL_UNSIGNED_INT, NULL);

  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glBindTexture(GL_TEXTURE_2D, 0);
  CheckOpenGLError("configure index texture");
  //std::cout << "opengl ok" << std::endl;

  glGenTextures(1, &m_normals_texture_id);
  CheckOpenGLError("generate normals texture");
  //std::cout << "opengl ok" << std::endl;

  glBindTexture(GL_TEXTURE_2D, m_normals_texture_id);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, m_size.x(), m_size.y(), 0, GL_LUMINANCE, GL_FLOAT, NULL);

  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glBindTexture(GL_TEXTURE_2D, 0);
  CheckOpenGLError("configure normals texture");

  // attach texture
  const GLenum color_attachment = GL_COLOR_ATTACHMENT0_EXT + 0;
  const GLenum depth_attachment = GL_COLOR_ATTACHMENT0_EXT + 1;
  const GLenum index_attachment = GL_COLOR_ATTACHMENT0_EXT + 2;
  const GLenum normals_attachment = GL_COLOR_ATTACHMENT0_EXT + 3;
  glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, m_framebuffer_id);
  glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, color_attachment, GL_TEXTURE_2D, m_color_texture_id, 0);
  glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, depth_attachment, GL_TEXTURE_2D, m_depth_texture_id, 0);
  glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, index_attachment, GL_TEXTURE_2D, m_index_texture_id, 0);
  glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, normals_attachment, GL_TEXTURE_2D, m_normals_texture_id, 0);
  glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
  CheckOpenGLError("attach color texture");
  //std::cout << "opengl ok" << std::endl;

  GLenum drawbuffers[4] = {color_attachment, depth_attachment, index_attachment, normals_attachment};

  glNamedFramebufferDrawBuffers(m_framebuffer_id, 4, drawbuffers);
  CheckOpenGLError("set drawbuffer list");

  glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, m_framebuffer_id);
  const GLenum completeness = glCheckFramebufferStatus(GL_FRAMEBUFFER_EXT);
  if (completeness != GL_FRAMEBUFFER_COMPLETE)
  {
    ROS_FATAL("surfel_next_best_view: error (%d): framebuffer is incomplete!", int(completeness));
    exit(1);
  }
  glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);

  //SHADER ROBOT
  m_program_robot_id = glCreateProgram();
  CheckOpenGLError("create robot program");

  m_vertex_shader_robot_id = glCreateShader(GL_VERTEX_SHADER);
  CheckOpenGLError("create robot vertex shader");
  const char * rvsource = VERTEX_SHADER_ROBOT.c_str();
  glShaderSource(m_vertex_shader_robot_id, 1, &rvsource, NULL);
  glCompileShader(m_vertex_shader_robot_id);
  CheckShaderError(m_vertex_shader_robot_id,"robot vertex shader");
  glAttachShader(m_program_robot_id, m_vertex_shader_robot_id);
  CheckOpenGLError("attach robot vertex shader");

  m_fragment_shader_robot_id = glCreateShader(GL_FRAGMENT_SHADER);
  CheckOpenGLError("create robot fragment shader");
  const std::string rfsourcestr = GetRobotFragmentShaderCode(m_with_color, m_with_normals);
  const char* rfsource = rfsourcestr.c_str();
  glShaderSource(m_fragment_shader_robot_id, 1, &rfsource, NULL);
  glCompileShader(m_fragment_shader_robot_id);
  CheckShaderError(m_fragment_shader_robot_id,"robot fragment shader");
  glAttachShader(m_program_robot_id, m_fragment_shader_robot_id);
  CheckOpenGLError("attach robot fragment shader");

  glLinkProgram(m_program_robot_id);
  CheckLinkError(m_program_robot_id,"render");

  m_maxdepth_robot_location = glGetUniformLocation(m_program_robot_id,"maxDepth");
  m_cam_robot_location = glGetUniformLocation(m_program_robot_id, "cam");
  m_t_inv_robot_location = glGetUniformLocation(m_program_robot_id, "t_inv");
  m_image_size_robot_location = glGetUniformLocation(m_program_robot_id, "image_size");
  CheckOpenGLError("get uniform location robot");
}

void RenderRobotURDF::CheckOpenGLError(const std::string message)
{
  static std::map<GLenum, std::string> errors;
  if (errors.empty())
  {
#define SurfelRenderer_CheckOpenGLError_ERROR_MACRO(a) \
  errors.insert(std::pair<GLenum, std::string>((a), (#a)));
    SurfelRenderer_CheckOpenGLError_ERROR_MACRO(GL_NO_ERROR);
    SurfelRenderer_CheckOpenGLError_ERROR_MACRO(GL_INVALID_ENUM);
    SurfelRenderer_CheckOpenGLError_ERROR_MACRO(GL_INVALID_VALUE);
    SurfelRenderer_CheckOpenGLError_ERROR_MACRO(GL_INVALID_OPERATION);
    SurfelRenderer_CheckOpenGLError_ERROR_MACRO(GL_STACK_OVERFLOW);
    SurfelRenderer_CheckOpenGLError_ERROR_MACRO(GL_STACK_UNDERFLOW);
    SurfelRenderer_CheckOpenGLError_ERROR_MACRO(GL_OUT_OF_MEMORY);
    SurfelRenderer_CheckOpenGLError_ERROR_MACRO(GL_TABLE_TOO_LARGE);
#undef SurfelRenderer_CheckOpenGLError_ERROR_MACRO
  }

  GLenum error = glGetError();

  if (error != GL_NO_ERROR)
  {
    auto iter = errors.find(error);
    if (iter != errors.end())
    {
      const char *es = iter->second.c_str();
      ROS_FATAL("surfel_next_best_view: openGL error: %s (%d)\nmessage: %s", es, int(error), message.c_str());
    }
    else
    {
      ROS_FATAL("surfel_next_best_view: openGL error: (unknown error) (%d)\nphase: %s", int(error), message.c_str());
    }
    exit(1);
  }
}

/****************************************************************************************************/

void RenderRobotURDF::CheckShaderError(const GLhandleARB shader_id, const std::string message)
{
  GLint status;
  glGetShaderiv(shader_id, GL_COMPILE_STATUS, &status);
  if (status != GL_TRUE)
  {
    const int SHADER_LOG_MAX_LEN = 10240;
    std::vector<char> infolog(SHADER_LOG_MAX_LEN + 1);
    GLsizei len;
    glGetShaderInfoLog(shader_id, SHADER_LOG_MAX_LEN, &len, infolog.data());
    infolog[len + 1] = '\0';
    ROS_FATAL("surfel_next_best_view: GLSL %s Shader compilation failed:\n%s\n\n", message.c_str(), infolog.data());
    exit(1);
  }
}

/****************************************************************************************************/

void RenderRobotURDF::CheckLinkError(const GLint program_id, const std::string message)
{
  GLint status;
  glGetProgramiv(program_id, GL_LINK_STATUS, &status);
  if (status != GL_TRUE)
  {
    const int PROGRAM_LOG_MAX_LEN = 10240;
    std::vector<char> infolog(PROGRAM_LOG_MAX_LEN + 1);
    GLsizei len;
    glGetProgramInfoLog(program_id, PROGRAM_LOG_MAX_LEN, &len, infolog.data());
    infolog[len + 1] = '\0';
    ROS_FATAL("surfel_next_best_view: GLSL %s Program link failed:\n%s\n\n", message.c_str(), infolog.data());
    exit(1);
  }
}


