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

#ifndef CNN_REAL_IMAGE_EVALUATE_H
#define CNN_REAL_IMAGE_EVALUATE_H

#define PARAM_NAME_MODE         "mode"
#define PARAM_VALUE_MODE_NONE   "none"
#define PARAM_VALUE_MODE_FIXED  "fixed"
#define PARAM_VALUE_MODE_ADHOC  "adhoc"
#define PARAM_VALUE_MODE_ONLY2D "only2d"
#define PARAM_VALUE_MODE_FULL   "projection_full"
#define PARAM_VALUE_MODE_GT     "gt"
#define PARAM_DEFAULT_MODE      PARAM_VALUE_MODE_NONE

#define PARAM_NAME_SCENARIO_FILE_PREFIX                  "scenario_file_prefix"
#define PARAM_DEFAULT_SCENARIO_FILE_PREFIX               ""

#define PARAM_NAME_SCENARIO_FIRST_INDEX                  "scenario_first_index"
#define PARAM_DEFAULT_SCENARIO_FIRST_INDEX               (int(0))

#define PARAM_NAME_SCENARIO_LAST_INDEX                   "scenario_last_index"
#define PARAM_DEFAULT_SCENARIO_LAST_INDEX                (int(1)) // last not included

#define PARAM_NAME_GT_FILE_PREFIX                        "gt_file_prefix"
#define PARAM_DEFAULT_GT_FILE_PREFIX                     ""

#define PARAM_NAME_VARIANT_FILE_PREFIX                   "environment_file_prefix"
#define PARAM_DEFAULT_VARIANT_FILE_PREFIX                ""

#define PARAM_NAME_MASK_FILE_NAME                        "mask_file_name"
#define PARAM_DEFAULT_MASK_FILE_NAME                     ""

#define PARAM_NAME_FIXED_MODE_FIXED_PROBABILITY          "fixed_mode_fixed_probability"
#define PARAM_DEFAULT_FIXED_MODE_FIXED_PROBABILITY       (double(0.15))

#define PARAM_NAME_IMAGE_FILE_PREFIX                     "image_file_prefix"
#define PARAM_DEFAULT_IMAGE_FILE_PREFIX                  ""

#define PARAM_NAME_CNN_BOUNDING_BOX_MIN                  "cnn_bounding_box_min"
#define PARAM_DEFAULT_CNN_BOUNDING_BOX_MIN               "0 0 0"

#define PARAM_NAME_CNN_BOUNDING_BOX_MAX                  "cnn_bounding_box_max"
#define PARAM_DEFAULT_CNN_BOUNDING_BOX_MAX               "1 1 1"

#define PARAM_NAME_CNN_BOUNDING_BOX_VOXEL_SIZE           "cnn_bounding_box_voxel_size"
#define PARAM_DEFAULT_CNN_BOUNDING_BOX_VOXEL_SIZE        (double(2.0)) // in voxelgrid voxels

#define PARAM_NAME_EVALUATION_FILE_PREFIX                "evaluation_file_prefix"
#define PARAM_DEFAULT_EVALUATION_FILE_PREFIX             ""

#define PARAM_NAME_WITH_SALIENCY_IMAGES                  "with_saliency_images"
#define PARAM_DEFAULT_WITH_SALIENCY_IMAGES               (bool(false))

#define PARAM_NAME_SAVE_IMAGES                           "save_image"
#define PARAM_DEFAULT_SAVE_IMAGES                        (bool(true))

#endif // CNN_REAL_IMAGE_EVALUATE_H
