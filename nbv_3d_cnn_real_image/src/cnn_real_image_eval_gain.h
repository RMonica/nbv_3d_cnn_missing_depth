#ifndef CNN_REAL_IMAGE_EVAL_GAIN_H
#define CNN_REAL_IMAGE_EVAL_GAIN_H

#define PARAM_NAME_MODE         "mode"
#define PARAM_VALUE_MODE_NONE   "none"
#define PARAM_VALUE_MODE_FIXED  "fixed"
#define PARAM_VALUE_MODE_ADHOC  "adhoc"
#define PARAM_VALUE_MODE_ONLY2D "only2d"
#define PARAM_VALUE_MODE_UNFROZEN  "unfrozen"
#define PARAM_VALUE_MODE_NOATTENTION  "noattention"
#define PARAM_VALUE_MODE_ONLY3D "only3d"
#define PARAM_VALUE_MODE_FULL   "projection_full"
#define PARAM_DEFAULT_MODE      PARAM_VALUE_MODE_NONE

#define PARAM_NAME_GT_EVAL_FILE_PREFIX                   "gt_evaluation_file_prefix"
#define PARAM_DEFAULT_GT_EVAL_FILE_PREFIX                ""

#define PARAM_NAME_SCENARIO_FILE_PREFIX                  "scenario_file_prefix"
#define PARAM_DEFAULT_SCENARIO_FILE_PREFIX               ""

#define PARAM_NAME_EVALUATION_FILE_PREFIX                "evaluation_file_prefix"
#define PARAM_DEFAULT_EVALUATION_FILE_PREFIX             ""

#define PARAM_NAME_EVAL_GAIN_FILE_PREFIX                 "eval_gain_file_prefix"
#define PARAM_DEFAULT_EVAL_GAIN_FILE_PREFIX              ""

#define PARAM_NAME_MASK_FILE_NAME                        "mask_file_name"
#define PARAM_DEFAULT_MASK_FILE_NAME                     ""

#define PARAM_NAME_SAVE_IMAGES                           "save_images"
#define PARAM_DEFAULT_SAVE_IMAGES                        (bool(true))

#define PARAM_NAME_SCENARIO_FIRST_INDEX                  "scenario_first_index"
#define PARAM_DEFAULT_SCENARIO_FIRST_INDEX               (int(0))

#define PARAM_NAME_SCENARIO_LAST_INDEX                   "scenario_last_index"
#define PARAM_DEFAULT_SCENARIO_LAST_INDEX                (int(1)) // last not included

#define PARAM_NAME_VARIANT_FILE_PREFIX                   "environment_file_prefix"
#define PARAM_DEFAULT_VARIANT_FILE_PREFIX                ""

#endif // CNN_REAL_IMAGE_EVAL_GAIN_H
