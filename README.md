# DepthAnything-ROS

`DepthAnything-ROS` is ROS2 wrapper for [Depth-Anything](https://github.com/LiheYoung/Depth-Anything).

https://github.com/scepter914/DepthAnything-ROS/assets/16330533/9fb5edd2-21b0-4441-a7c1-b304246e49b5

- Environment
  - Ubuntu 22.04.01, ROS2 Humble
  - CUDA 12.3, cuDNN 8.9.5.29-1+cuda12.2, TensorRT 8.6.1.6-1+cuda12.0

## Get started
### Set environment

- Install ROS2

See [ROS2 document](https://docs.ros.org/en/humble/Installation.html).

To install ROS2 easily, I recommend to use ansible script of [Autoware](https://github.com/autowarefoundation/autoware).
In detail, please see [the installation page](https://autowarefoundation.github.io/autoware-documentation/main/installation/autoware/source-installation/).

- Install dependency

```
sudo apt install libgflags-dev libboost-all-dev
```

- Prepare your rosbag

If you don't have any rosbag, I recommend [rosbag for Nuscenes dataset](https://github.com/scepter914/nuscenes_rosbag).

### Set onnx files

Set onnx files for `DepthAnything-ROS/data` or set `onnx_path` parameter.

```
  <arg name="onnx_path" default="$(find-pkg-share depth_anything)/data/depth_anything_vitb14.onnx" />
```

Run below command to get the onnx files of pre-train model.

```sh
# Install gdown
pip install gdown
# Download onnx file
mkdir data && cd data
gdown 1jFTCJv0uJovPAww9PHCYAoek-KfeajK_
```

If you want to make onnx files at yourself, please use [depth-anything-tensorrt](https://github.com/spacewalk01/depth-anything-tensorrt).

### Launch

```
ros2 launch depth_anything depth_anything.launch.xml
```

## Interface
### Input

- `input/image` (`sensor_msgs::msg::Image`)

The input image.

### Output

- `~/output/depth_image` (`sensor_msgs::msg::Image`)

The depth image made by DepthAnything.

### Parameters

- `onnx_path` (string)
  - Default parameter: "$(find-pkg-share depth_anything)/data/depth_anything_vitb14.onnx"

The path to onnx file.

- `precision` (string)
  - Default parameter: "fp32"

The precision mode to use quantization.
DepthAnything-ROS supports in "fp32" or "fp16" ([#2](https://github.com/scepter914/DepthAnything-ROS/issues/2)) for now.

## Note
### Build for TensorRT engine

When you run on the first start up, you need to wait about 5 minutes for build step.

### Performance

- Performance
  - RTX4090 results is written in official code

| Model                | Params | RTX4090 TensorRT |  RTX2070 TensorRT   |
| :------------------- | -----: | :--------------: | :-----------------: |
| Depth-Anything-Small |  24.8M |       3 ms       |  27 ms, VRAM 300MB  |
| Depth-Anything-Base  |  97.5M |       6 ms       |  65 ms, VRAM 700MB  |
| Depth-Anything-Large | 335.3M |      12 ms       | 200 ms, VRAM 1750MB |

### Reference

- [Depth-Anything](https://github.com/LiheYoung/Depth-Anything)
- [trt-depth-anything](https://github.com/daniel89710/trt-depth-anything) Use for TensorRT inference
- [depth-anything-tensorrt](https://github.com/spacewalk01/depth-anything-tensorrt) Use for making from pth files to onnx files
- [tensorrt_common](https://github.com/autowarefoundation/autoware.universe/tree/main/common/tensorrt_common) Use library
- [cuda_utils](https://github.com/autowarefoundation/autoware.universe/tree/main/common/cuda_utils) Use library
- [gdown](https://github.com/wkentaro/gdown)
