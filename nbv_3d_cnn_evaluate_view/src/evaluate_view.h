#ifndef NBV_3D_CNN_EVALUATE_VIEW_H
#define NBV_3D_CNN_EVALUATE_VIEW_H

// --------- MAIN PARAMETERS ------------

#define PARAM_NAME_AP_LOST_IF_OCCUPIED            "a_priori_ray_lost_if_occupied"
#define PARAM_DEFAULT_AP_LOST_IF_OCCUPIED         (double(0.15))

// default occupancy probability if the ray is out of the scanned bounding box
#define PARAM_NAME_AP_OCCUPIED_IF_OUTSIDE         "a_priori_ray_occupied_if_outside"
#define PARAM_DEFAULT_AP_OCCUPIED_IF_OUTSIDE      (double(0.001))

// default occupancy probability if the ray is inside the bounding box, but in unknown space
#define PARAM_NAME_AP_OCCUPIED                    "a_priori_occupancy_probability"
#define PARAM_DEFAULT_AP_OCCUPIED                 (double(0.001))

#define PARAM_NAME_SET_ENVIRONMENT_ACTION_NAME    "set_environment_action_name"
#define PARAM_DEFAULT_SET_ENVIRONMENT_ACTION_NAME "set_environment"

#define PARAM_NAME_ENVIRONMENT_ORIGIN_OFFSET      "environment_origin_offset"
#define PARAM_DEFAULT_ENVIRONMENT_ORIGIN_OFFSET   (double(0)) // in voxel sizes

#define PARAM_NAME_EVALUATE_VIEW_ACTION_NAME      "evaluate_view_action_name"
#define PARAM_DEFAULT_EVALUATE_VIEW_ACTION_NAME   "evaluate_view"

#define PARAM_NAME_RAW_PROJECTION_ACTION_NAME     "raw_projection_action_name"
#define PARAM_DEFAULT_RAW_PROJECTION_ACTION_NAME  "raw_projection"

#define PARAM_NAME_GROUND_TRUTH_EV_ACTION_NAME    "ground_truth_evaluate_view_action_name"
#define PARAM_DEFAULT_GROUND_TRUTH_EV_ACTION_NAME "ground_truth_evaluate_view"

#define PARAM_NAME_CIRCLE_FILTER_ACTION_NAME      "generate_circle_filter_mask_action_name"
#define PARAM_DEFAULT_CIRCLE_FILTER_ACTION_NAME   "generate_filter_circle_rectangle_mask"

#define PARAM_NAME_RENDER_ROBOT_URDF_SERVICE      "render_robot_urdf_service"
#define PARAM_DEFAULT_RENDER_ROBOT_URDF_SERVICE   "/render_robot_urdf/render_robot_urdf"

// --------- FILTER PARAMETERS -------------

#define PARAM_NAME_FILTER_CIRCLE_RADIUS      "filter_circle_radius"
#define PARAM_DEFAULT_FILTER_CIRCLE_RADIUS   (double(0.0))

#define PARAM_NAME_FILTER_CIRCLE_CENTER      "filter_circle_center"
#define PARAM_DEFAULT_FILTER_CIRCLE_CENTER   "0.0 0.0"

#define PARAM_NAME_FILTER_RECTANGLE_WIDTH    "filter_rectangle_width"
#define PARAM_DEFAULT_FILTER_RECTANGLE_WIDTH (double(1.0))

#define PARAM_NAME_FILTER_RECTANGLE_HEIGHT   "filter_rectangle_height"
#define PARAM_DEFAULT_FILTER_RECTANGLE_HEIGHT (double(1.0))

#define PARAM_NAME_FILTER_RECTANGLE_X        "filter_rectangle_x"
#define PARAM_DEFAULT_FILTER_RECTANGLE_X     (double(0.0))

#define PARAM_NAME_FILTER_RECTANGLE_Y        "filter_rectangle_y"
#define PARAM_DEFAULT_FILTER_RECTANGLE_Y     (double(0.0))

#define PARAM_NAME_FILTER_DISCONTINUITY_TH   "filter_discontinuity_th"
#define PARAM_DEFAULT_FILTER_DISCONTINUITY_TH (double(0.01))

#define PARAM_NAME_FILTER_DISCONTINUITY_WIN  "filter_discontinuity_window"
#define PARAM_DEFAULT_FILTER_DISCONTINUITY_WIN (int(1))

#define PARAM_NAME_FILTER_PIXEL_SHIFT_X      "filter_pixel_shift_x"
#define PARAM_DEFAULT_FILTER_PIXEL_SHIFT_X   (int(0))

#define PARAM_NAME_FILTER_PIXEL_SHIFT_Y      "filter_pixel_shift_y"
#define PARAM_DEFAULT_FILTER_PIXEL_SHIFT_Y   (int(0))

#define PARAM_NAME_FILTER_OPENING_SIZE       "filter_opening_size"
#define PARAM_DEFAULT_FILTER_OPENING_SIZE    (int(0))

#define PARAM_NAME_FILTER_EROSION_SIZE       "filter_erosion_size"
#define PARAM_DEFAULT_FILTER_EROSION_SIZE    (int(0))

#define PARAM_NAME_FILTER_EROSION_DEPTH_SCALE "filter_erosion_depth_scale"
#define PARAM_DEFAULT_FILTER_EROSION_DEPTH_SCALE (double(0.5))

#define PARAM_NAME_FILTER_NORMAL_MAX_ANGLE   "normal_max_angle"
#define PARAM_DEFAULT_FILTER_NORMAL_MAX_ANGLE (double(70.0)) // degrees

#define PARAM_NAME_SHADOW_REMOVAL_EMITTER_DISTANCE_RIGHT    "shadow_removal_emitter_distance_right"
#define PARAM_DEFAULT_SHADOW_REMOVAL_EMITTER_DISTANCE_RIGHT (double(0.0))

#define PARAM_NAME_SHADOW_REMOVAL_EMITTER_DISTANCE_LEFT     "shadow_removal_emitter_distance_left"
#define PARAM_DEFAULT_SHADOW_REMOVAL_EMITTER_DISTANCE_LEFT  (double(0.0))

// ------ PREDICT ACTION -----

#define PARAM_NAME_PREDICT_AUTOCOMPLETE_ACTION_NAME    "predict_autocomplete_action_name"
#define PARAM_DEFAULT_PREDICT_AUTOCOMPLETE_ACTION_NAME "predict" // set to empty to disable

#define PARAM_NAME_PREDICT_IMAGE_ACTION_NAME           "predict_image_action_name"
#define PARAM_DEFAULT_PREDICT_IMAGE_ACTION_NAME        "" // set to empty to disable

#define PARAM_NAME_PREDICT_PROJECTION_ACTION_NAME      "predict_projection_action_name"
#define PARAM_DEFAULT_PREDICT_PROJECTION_ACTION_NAME   "" // set to empty to disable

#define PARAM_NAME_PIP_CROP_BBOX_MIN            "crop_bbox_min"
#define PARAM_DEFAULT_PIP_CROP_BBOX_MIN         "0 0 0"

#define PARAM_NAME_PIP_CROP_BBOX_MAX            "crop_bbox_max"
#define PARAM_DEFAULT_PIP_CROP_BBOX_MAX         "1 1 1"

#define PARAM_NAME_CNN_MAX_RANGE                "cnn_max_range"
#define PARAM_DEFAULT_CNN_MAX_RANGE             (double(1.5))

#define PARAM_NAME_CNN_IMAGE_DEPTH              "cnn_image_depth"
#define PARAM_DEFAULT_CNN_IMAGE_DEPTH           (int(64))

#endif // NBV_3D_CNN_EVALUATE_VIEW_H
