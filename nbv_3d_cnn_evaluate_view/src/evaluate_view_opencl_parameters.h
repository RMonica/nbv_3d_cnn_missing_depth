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
