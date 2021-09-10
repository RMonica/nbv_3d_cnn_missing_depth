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
