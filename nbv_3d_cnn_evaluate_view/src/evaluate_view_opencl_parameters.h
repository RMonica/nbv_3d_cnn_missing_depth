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

#ifndef EVALUATE_VIEW_OPENCL_PARAMETERS_H
#define EVALUATE_VIEW_OPENCL_PARAMETERS_H

#define PARAM_NAME_OPENCL_PLATFORM_NAME        "opencl_platform_name"
#define PARAM_DEFAULT_OPENCL_PLATFORM_NAME     ""           // a part of the name is enough, empty = default

#define PARAM_NAME_OPENCL_DEVICE_NAME          "opencl_device_name"
#define PARAM_DEFAULT_OPENCL_DEVICE_NAME       ""           // a part of the name is enough, empty = default

#define PARAM_NAME_OPENCL_DEVICE_TYPE          "opencl_device_type"
#define PARAM_VALUE_OPENCL_DEVICE_TYPE_GPU     "GPU"
#define PARAM_VALUE_OPENCL_DEVICE_TYPE_CPU     "CPU"
#define PARAM_VALUE_OPENCL_DEVICE_TYPE_ALL     "ALL"
#define PARAM_DEFAULT_OPENCL_DEVICE_TYPE       PARAM_VALUE_OPENCL_DEVICE_TYPE_ALL

#endif // EVALUATE_VIEW_OPENCL_PARAMETERS_H
