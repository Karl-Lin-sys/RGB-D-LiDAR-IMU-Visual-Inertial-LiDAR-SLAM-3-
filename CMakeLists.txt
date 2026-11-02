cmake_minimum_required(VERSION 3.18)
project(vil_semantic_slam LANGUAGES CXX)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 20)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -O3)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)
find_package(PCL REQUIRED COMPONENTS common filters kdtree)
find_package(OpenCV REQUIRED COMPONENTS core imgproc)

include_directories(
  include
  ${EIGEN3_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

add_executable(vil_semantic_slam_node src/main.cpp)
ament_target_dependencies(vil_semantic_slam_node
  rclcpp
  sensor_msgs
  pcl_conversions
)
target_link_libraries(vil_semantic_slam_node
  Eigen3::Eigen
  Ceres::ceres
  ${PCL_LIBRARIES}
  ${OpenCV_LIBS}
)

install(TARGETS vil_semantic_slam_node
  DESTINATION lib/${PROJECT_NAME}
)
install(DIRECTORY include/
  DESTINATION include/
)

ament_package()
