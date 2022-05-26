nbv_3d_cnn_missing_depth
========================

Deep learning method to predict the missing measurements of a virtual depth camera, given a representation of the environment and the 3D pose of the depth camera.<br />
That is, given a 3D pose of a virtual camera and a ternary voxel grid of the environment, this code predicts which pixels would contain missing measurements if the camera was actually placed at the given pose.<br />
The result may be used to improve prediction of the information gain of candidate views in Next Best View planning.

Related publication:

- R. Monica, J. Aleotti, *Prediction of Depth Camera Missing Measurements Using Deep Learning for Next Best View Planning*, IEEE International Conference on Robotics and Automation (ICRA), May 23 - 27, 2022, Philadelphia (USA)

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

### Dataset

The dataset is composed of multiple complete 3D reconstructions of (tabletop) scenes, and the corresponding ground truth depth images acquired from various poses using the real depth camera.

The dataset must be placed into folder `nbv_3d_cnn_real_image/data/training_images`. One sub-folder for each scene, numbered incrementally: `scenario_1`, `scenario_2`, etc.<br/>
Each scene contains:

- `kinfu_voxelgrid.voxelgrid`: the raw 3D voxel grid of the environment, in the format supported by `rmonica_voxelgrid_common`.
- `voxelgrid_metadata.txt`: a text file defining 3D position and voxel size of the voxelgrid.
- `poi.txt`: a text file containing the definition of a sphere in 3D space, as CenterX CenterY CenterZ Radius. The sphere should be centered on the interesting part of the scene, because parts of this sphere will be set to unknown during partial environment generation.
- `images`: a sub-folder, containing, for each ground truth image:
    - `camera_info_X.txt`: camera intrinsic parameters (serialized camera info message).
    - `depth_image_X_Y.png`: 16-bit depth image.
    - `pose_X.matrix`: text file containing the 3D camera pose, as a 4x4 matrix.
    - `rgb_image_X_Y.png`: corresponding color image (unused).
    - `joint_state_X.txt`: robot joint state.<br />
    where X is the pose number, 0-based, and Y is the image number (multiple images can be taken from the same pose).
- `partial_environments`: empty folder, which will be filled with the generated partial environments.
- `virtual_images`: empty folder, which will be filled with the generated ground truth images and input images.

Some sample scenes may be downloaded from here:
[scenario_1](http://rimlab.ce.unipr.it/~rmonica/nbv_3d_cnn_real_image_dataset_scenario1.zip)
[scenario_2](http://rimlab.ce.unipr.it/~rmonica/nbv_3d_cnn_real_image_dataset_scenario2.zip)
[scenario_3](http://rimlab.ce.unipr.it/~rmonica/nbv_3d_cnn_real_image_dataset_scenario3.zip)
[scenario_4](http://rimlab.ce.unipr.it/~rmonica/nbv_3d_cnn_real_image_dataset_scenario4.zip)
(650-700 MB each).

### Dataset pre-processing

For each scene:

- Edit `nbv_3d_cnn_real_image/launch/generate_partial_environments.launch` and `nbv_3d_cnn_real_image/launch/generate_virtual_views.launch`, set arg `scenario` to the scene number.
- Launch:

    ```roslaunch nbv_3d_cnn_real_image generate_partial_environments.launch```

- Launch:

    ```roslaunch nbv_3d_cnn_real_image generate_virtual_views.launch```

### Training the auto-completion network

First, the auto-completion network in the `nbv_3d_prob_cnn` repository must be trained, as its weights are used by the main network for transfer learning. The auto-completion network should be re-trained from this repository (it uses slightly different parameters).

The dataset must be placed into folder `nbv_3d_cnn_real_image/data/environments_3d_realistic`. The dataset is composed of partially-known scenes. For each scene, four voxelgrids must be provided:

- `X_empty.binvoxelgrid`: voxelgrid with value `1` where the voxel is known and empty, `0` otherwise.
- `X_occupied.binvoxelgrid`: voxelgrid with value `1` where the voxel is known and occupied, `0` otherwise.
- `X_unknown.binvoxelgrid`: voxelgrid with value `1` where the voxel is unknown, `0` otherwise.
- `X_environment.binvoxelgrid`: ground truth voxelgrid, with value `1` where the environment is actually occupied and `0` otherwise (no unknown values).

where `X` is the scene number.

All voxelgrids are in the format loadable by the `rmonica_voxelgrid_common` package. A suitable dataset may be downloaded from [here](http://rimlab.ce.unipr.it/~rmonica/nbv_3d_cnn_real_image_environments_3d_realistic.zip). It also contains `.bt` OctoMap files for visualization.<br />
**Warning:** the download is less than 30 MB, but it is over 4 GB when uncompressed.

In the launch file `nbv_3d_cnn_real_image/launch/cnn_real_image_autocomplete_train.launch`, these parameters can be used to split the dataset into training set and test set:

- `training_dataset_first_element`: first scenario used for training.
- `training_dataset_last_element`: one past the last scenario used for training.
- `validation_dataset_first_element`: first scenario used for testing.
- `validation_dataset_last_element`: one past the last scenario used for testing.

For training, launch the file:
```
  roslaunch nbv_3d_cnn_real_image cnn_real_image_autocomplete_train.launch
```

Output is written into the folder `nbv_3d_cnn_real_image/data/output_autocomplete`. Create the folder beforehand if not existing.

### Training the main network

The package supports training for the main network proposed in the paper, and the four other networks for the ablation study.

- **projection**: the method proposed in the paper.
- **only2d**: called "w/o 3D" in the paper.
- **only3d**: called "w/o imgs" in the paper.
- **noattention**: called "w/o attention" in the paper.
- **unfrozen**: called "w/o frozen 3D" in the paper.

Training of `projection`, `only3d`, `noattention` and `unfrozen` is done using the launch file `cnn_real_image_projection_train.launch`. Select the method by setting arg `mode` at the beginning of the launch file: 
```
  roslaunch nbv_3d_cnn_real_image cnn_real_image_projection_train.launch
```

Training of `only2d` is done using the launch file `cnn_real_image_model_train.launch`:
```
  roslaunch nbv_3d_cnn_real_image cnn_real_image_model_train.launch
```

In each launch file, these parameters can be used to split the dataset into training set and test set:

- `training_dataset_first_element`: first scenario used for training.
- `training_dataset_last_element`: one past the last scenario used for training.
- `validation_dataset_first_element`: first scenario used for testing.
- `validation_dataset_last_element`: one past the last scenario used for testing.

At the end of the training, output images and trained model will be written into the folder:
```
  nbv_3d_cnn_real_image/data/output_METHOD
```
where `METHOD` is the network name (e.g., `only2d`).<br />
Create the folder beforehand.

During training, the `nbv_3d_cnn_real_image/data/tensorboard` folder will be used for tensorboard.

### Evaluation

To evaluate one of the methods, launch:
```
  nbv_3d_cnn_real_image/launch/cnn_real_image_evaluate.launch
```

Note that a trained auto-completion model (from `nbv_3d_cnn`) is required. Configure parameter `checkpoint_file` of node `nbv_3d_cnn_predict.py` to load the proper file.

Set the arg `mode` to the method to be evaluated. In addition to the networks (**projection**, **only2d**, **only3d**, **noattention**, **unfrozen**), these other methods can be evaluated:

- **fixed**: fixed valid measurement probability for the whole depth image. The value can be set by changing the parameter `fixed_mode_fixed_probability`.
- **none**: valid measurement probability set to `1` for the whole depth image.
- **gt**: ground truth generation. Predicts the correct information gain by accessing the ground truth valid measurement masks.
- **adhoc**: currently broken, do not use.

Evaluation output is written into the folder configured by the `evaluation_file_prefix` parameter. All valid measurement probability maps are saved.<br />
Moreover, these files are generated:

- `aaa_log.csv`: for each view, the information gain predicted by the method.
- `aaa_timers.csv`: for each view, the method computation time.

2022-05-26
