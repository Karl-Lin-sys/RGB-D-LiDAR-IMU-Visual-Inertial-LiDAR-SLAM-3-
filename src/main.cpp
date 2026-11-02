// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Karl-Lin-sys & Contributors

#include "vil_semantic_slam/vil_semantic_slam_node.hpp"

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::executors::MultiThreadedExecutor executor;
  auto node = std::make_shared<vil_slam::VilSemanticSlamNode>(rclcpp::NodeOptions{});
  executor.add_node(node);
  executor.spin();
  rclcpp::shutdown();
  return 0;
}
