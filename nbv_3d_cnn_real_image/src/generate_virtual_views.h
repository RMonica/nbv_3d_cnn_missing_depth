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
