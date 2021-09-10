/*
 * Copyright (c) 2018, Francesco Periti, Riccardo Monica
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

#include "surfel_fragment_shader_robot.h"

static const std::string FRAGMENT_SHADER = R"763d539980(

uniform float maxDepth;

in vec4 position;
flat in vec3 normal;
in float delete_me;

layout(location = 0) out vec4 image;
layout(location = 1) out float depth_image;
layout(location = 2) out int index_image;
layout(location = 3) out vec3 normal_image;

void main()
{
  #if WITH_COLOR
  image.xyz = vec3(1.0, 0.0, 0.0);
  image.xyz = image.xyz * abs(dot(normalize(normal), normalize(position.xyz)));
  image.w = 1.0;
  #endif // WITH_COLOR

  #if WITH_NORMALS
  normal_image = normalize(normal);
  #endif

  depth_image = position.z;

  #if 0
  index_image = 0;
  #endif

  if(position.z > maxDepth)
    discard;
      
  if (position.z <= 0.0)
    discard;

  if (delete_me > 0.0001)
    discard;

  gl_FragDepth = (position.z / (2 * maxDepth)) + 0.5f;
}

                                          )763d539980";

std::string GetRobotFragmentShaderCode(const bool with_color,
                                       const bool with_normals)
{
  std::string result;
  result += "#version 330 compatibility\n";

  result += "#define WITH_COLOR ";
  result += (with_color ? "1" : "0");
  result += "\n";

  result += "#define WITH_NORMALS ";
  result += (with_normals ? "1" : "0");
  result += "\n";

  result += FRAGMENT_SHADER;

  return result;
}
