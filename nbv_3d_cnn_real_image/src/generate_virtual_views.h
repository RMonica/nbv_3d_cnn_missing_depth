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

#ifndef GENERATE_VIRTUAL_VIEWS_H
#define GENERATE_VIRTUAL_VIEWS_H

#define PARAM_NAME_VOXELGRID_PREFIX             "voxelgrid_prefix"
#define PARAM_DEFAULT_VOXELGRID_PREFIX          ""

#define PARAM_NAME_TSDF_VOLUME_FILENAME         "tsdf_volume_filename"
#define PARAM_DEFAULT_TSDF_VOLUME_FILENAME      ""

#define PARAM_NAME_VOXELGRID_METADATA_FILENAME  "voxelgrid_metadata_filename"
#define PARAM_DEFAULT_VOXELGRID_METADATA_FILENAME ""

#define PARAM_NAME_INPUT_IMAGE_PREFIX           "input_image_prefix"
#define PARAM_DEFAULT_INPUT_IMAGE_PREFIX        ""

#define PARAM_NAME_OUTPUT_IMAGE_PREFIX          "output_image_prefix"
#define PARAM_DEFAULT_OUTPUT_IMAGE_PREFIX       ""

#define PARAM_NAME_IMAGE_NUMBER                 "image_number"
#define PARAM_DEFAULT_IMAGE_NUMBER              (int(0))

#define PARAM_NAME_IMAGE_NUMBER_COUNT           "image_number_count"
#define PARAM_DEFAULT_IMAGE_NUMBER_COUNT        (int(1))

#define PARAM_NAME_DEPTH_IMAGES_PER_POSE        "depth_images_per_pose"
#define PARAM_DEFAULT_DEPTH_IMAGES_PER_POSE     (int(5))

#define PARAM_NAME_MAX_RANGE                    "max_range"
#define PARAM_DEFAULT_MAX_RANGE                 (double(4.0))

#define PARAM_NAME_MIN_RANGE                    "min_range"
#define PARAM_DEFAULT_MIN_RANGE                 (double(0.5))

#define PARAM_NAME_POI_FILE_NAME                "poi_file_name"
#define PARAM_DEFAULT_POI_FILE_NAME             ""

#endif // GENERATE_VIRTUAL_VIEWS_H
