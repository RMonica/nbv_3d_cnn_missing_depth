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
