nbv_3d_cnn_missing_depth
========================

Prediction of depth camera missing measurements given a ternary voxel grid and a 3D pose.

Installation
------------

This repository contains ROS (Robot Operating System) packages.
Download the repository into your workspace and compile it with `catkin build` or `catkin_make`.

### System dependencies:

- ROS (Noetic)
- OpenCV
- Eigen3
- Point Cloud Library
- OpenCL
- OpenGL
- TensorFlow

### ROS dependencies:

- [Message Serialization](https://github.com/swri-robotics/message_serialization)
- [Init Fake OpenGL Context](https://github.com/RMonica/init_fake_opengl_context)
- [nbv_3d_prob_cnn](https://github.com/RMonica/nbv_3d_prob_cnn)
- [rmonica_voxelgrid_common](https://github.com/RMonica/rmonica_voxelgrid_common)

**Note**<br/>
The package `render_robot_urdf` requires the CAD models of the COMAU Smart Six robot manipulator. We are not allowed to distribute these CAD models. Simplified meshes have been provided in `render_robot_urdf/meshes/stl/convex_hull`. To use them, please change references from `comau_nolimits_nobase.urdf` into `comau_nolimits_nobase_chull.urdf` in the launch files.

**Note**<br/>
By default, ROS compiles without optimizations and produces very slow executables. Please activate optimizations. Example commands:

```
  catkin_make -DCMAKE_BUILD_TYPE=RelWithDebInfo
  catkin build --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo
```


Usage
-----

TODO

