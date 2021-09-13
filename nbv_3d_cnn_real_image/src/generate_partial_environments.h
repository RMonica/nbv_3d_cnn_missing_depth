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

#ifndef GENERATE_PARTIAL_ENVIRONMENTS_H
#define GENERATE_PARTIAL_ENVIRONMENTS_H

#define PARAM_NAME_VOXELGRID_FILENAME       "voxelgrid_filename"
#define PARAM_DEFAULT_VOXELGRID_FILENAME    ""

#define PARAM_NAME_METADATA_FILENAME        "metadata_filename"
#define PARAM_DEFAULT_METADATA_FILENAME     ""

#define PARAM_NAME_OUTPUT_PREFIX            "output_prefix"
#define PARAM_DEFAULT_OUTPUT_PREFIX         ""

#define PARAM_NAME_IMAGES_PREFIX            "images_prefix"
#define PARAM_DEFAULT_IMAGES_PREFIX         ""

#define PARAM_NAME_CROP_BBOX_MIN            "crop_bbox_min"
#define PARAM_DEFAULT_CROP_BBOX_MIN         "0 0 0"

#define PARAM_NAME_CROP_BBOX_MAX            "crop_bbox_max"
#define PARAM_DEFAULT_CROP_BBOX_MAX         "1 1 1"

#define PARAM_NAME_POI_FILE_NAME            "poi_file_name"
#define PARAM_DEFAULT_POI_FILE_NAME         ""

#define PARAM_NAME_SELECT_VIEWPOINTS_MIN    "select_viewpoints_min"
#define PARAM_DEFAULT_SELECT_VIEWPOINTS_MIN (int(1))

#define PARAM_NAME_SELECT_VIEWPOINTS_MAX    "select_viewpoints_max"
#define PARAM_DEFAULT_SELECT_VIEWPOINTS_MAX (int(1))

#define PARAM_NAME_NUM_INCOMPLETE_ENV       "num_incomplete_environments"
#define PARAM_DEFAULT_NUM_INCOMPLETE_ENV    (int(1))

#endif // GENERATE_PARTIAL_ENVIRONMENTS_H
