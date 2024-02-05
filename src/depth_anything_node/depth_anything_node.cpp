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

#include "depth_anything/depth_anything_node.hpp"

#include <memory>
#include <string>
#include <vector>

namespace
{
template <class T>
bool update_param(
  const std::vector<rclcpp::Parameter> & params, const std::string & name, T & value)
{
  const auto itr = std::find_if(
    params.cbegin(), params.cend(),
    [&name](const rclcpp::Parameter & p) { return p.get_name() == name; });

  // Not found
  if (itr == params.cend()) {
    return false;
  }

  value = itr->template get_value<T>();
  return true;
}
}  // namespace

namespace depth_anything
{
using namespace std::literals;

DepthAnythingNode::DepthAnythingNode(const rclcpp::NodeOptions & node_options)
: Node("depth_anything", node_options)
{
  using std::placeholders::_1;
  // Parameter
  set_param_res_ =
    this->add_on_set_parameters_callback(std::bind(&DepthAnythingNode::onSetParam, this, _1));
  node_param_.onnx_path = declare_parameter<std::string>("onnx_path");

  std::cout << node_param_.onnx_path << std::endl;
  node_param_.precision = declare_parameter<std::string>("precision");

  // Subscriber
  sub_image_ = image_transport::create_subscription(
    this, "~/input/image", std::bind(&DepthAnythingNode::onData, this, _1), "raw",
    rmw_qos_profile_sensor_data);

  // Publisher
  pub_depth_image_ = create_publisher<sensor_msgs::msg::Image>("~/output/depth_image", 1);

  // Init TensorRT model
  std::string calibType = "MinMax";
  int dla = -1;
  bool first = false;
  bool last = false;
  bool prof = false;
  double clip = 0.0;
  tensorrt_common::BuildConfig build_config(calibType, dla, first, last, prof, clip);

  int batch = 1;
  tensorrt_common::BatchConfig batch_config{1, batch / 2, batch};

  bool use_gpu_preprocess = false;
  std::string calibration_images = "calibration_images.txt";
  const size_t workspace_size = (1 << 30);

  trt_depth_anything = std::make_unique<tensorrt_depth_anything::TrtDepth_Anything>(
    node_param_.onnx_path, node_param_.precision, build_config, use_gpu_preprocess,
    calibration_images, batch_config, workspace_size);
  RCLCPP_INFO(get_logger(), "Finish initialize TensorRT model");
}

void DepthAnythingNode::onData(const sensor_msgs::msg::Image::ConstSharedPtr msg)
{
  cv_bridge::CvImagePtr in_image_ptr;
  try {
    in_image_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
  } catch (cv_bridge::Exception & e) {
    RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
    return;
  }
  const auto width = in_image_ptr->image.cols;
  const auto height = in_image_ptr->image.rows;

  if (is_initialized == false) {
    trt_depth_anything->initPreprocessBuffer(width, height);
    is_initialized = true;
  }

  std::vector<cv::Mat> input_images;
  input_images.push_back(in_image_ptr->image);
  trt_depth_anything->doInference(input_images);
  cv::Mat depth_image = trt_depth_anything->getDepthImage();
  const auto scale_size = cv::Size(width, height);
  cv::resize(depth_image, depth_image, scale_size, 0, 0, cv::INTER_NEAREST);

  cv_bridge::CvImage cv_img;
  cv_img.image = depth_image;
  cv_img.encoding = "mono8";

  sensor_msgs::msg::Image depth_image_msgs;
  cv_img.toImageMsg(depth_image_msgs);
  depth_image_msgs.header = msg->header;
  pub_depth_image_->publish(depth_image_msgs);
}

rcl_interfaces::msg::SetParametersResult DepthAnythingNode::onSetParam(
  const std::vector<rclcpp::Parameter> & params)
{
  rcl_interfaces::msg::SetParametersResult result;
  try {
    {
      auto & p = node_param_;
      update_param(params, "onnx_path", p.onnx_path);
      update_param(params, "precision", p.precision);
    }
  } catch (const rclcpp::exceptions::InvalidParameterTypeException & e) {
    result.successful = false;
    result.reason = e.what();
    return result;
  }
  result.successful = true;
  result.reason = "success";
  return result;
}

bool DepthAnythingNode::isDataReady()
{
  if (!image_data_) {
    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000, "waiting for data msg...");
    return false;
  }
  return true;
}

}  // namespace depth_anything

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(depth_anything::DepthAnythingNode)
