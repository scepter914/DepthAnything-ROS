// Copyright 2024 Satoshi Tanaka
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef DEPTH_ANYTHING__DEPTH_ANYTHING_NODE_HPP__
#define DEPTH_ANYTHING__DEPTH_ANYTHING_NODE_HPP__

#if __has_include(<cv_bridge/cv_bridge.hpp>)
#include <cv_bridge/cv_bridge.hpp>
#else
#include <cv_bridge/cv_bridge.h>
#endif

#include <image_transport/image_transport.hpp>
#include <memory>
#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>
#include <string>

#include "tensorrt_depth_anything/tensorrt_depth_anything.hpp"

namespace depth_anything
{
class DepthAnythingNode : public rclcpp::Node
{
public:
  explicit DepthAnythingNode(const rclcpp::NodeOptions & node_options);

  struct NodeParam
  {
    std::string onnx_path{};
    std::string precision{};
  };

private:
  // Subscriber
  image_transport::Subscriber sub_image_;

  // Callback
  void onData(const sensor_msgs::msg::Image::ConstSharedPtr msg);

  // Data Buffer
  sensor_msgs::msg::Image::ConstSharedPtr image_data_{};

  // Publisher
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_depth_image_;

  // Timer
  bool isDataReady();

  // Parameter Server
  OnSetParametersCallbackHandle::SharedPtr set_param_res_;
  rcl_interfaces::msg::SetParametersResult onSetParam(
    const std::vector<rclcpp::Parameter> & params);

  // Parameter
  NodeParam node_param_{};

  // Core

  std::shared_ptr<tensorrt_depth_anything::TrtDepth_Anything> trt_depth_anything;
  bool is_initialized = false;
};

}  // namespace depth_anything

#endif  // DEPTH_ANYTHING__DEPTH_ANYTHING_NODE_HPP__
