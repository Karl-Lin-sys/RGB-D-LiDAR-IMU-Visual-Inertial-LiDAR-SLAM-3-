// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Karl-Lin-sys & Contributors
// Compliant with Taiwan (R.O.C.) Copyright Act & Trade Secrets Act.
// Built under NIST Cybersecurity Framework (CSF) 2.0 Guidelines.

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

// ROS 2 Official C++ API (rclcpp)
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl_conversions/pcl_conversions.h>

// Eigen3 Official C++ API
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>

// Ceres Solver Official C++ API
#include <ceres/ceres.h>
#include <ceres/rotation.h>

// PCL Official C++ API
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>

// OpenCV Official C++ API
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#ifdef ENABLE_TENSORRT
// NVIDIA TensorRT Official C++ API (v10+) & CUDA Runtime
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#endif

namespace vil_slam {

#ifdef ENABLE_TENSORRT
// ============================================================================
// [NIST CSF: PROTECT (PR.DS)] RAII Custom Deleters for GPU / TensorRT Resources
// Prevents memory leaks, dangling pointers, and unfreed VRAM on exceptions.
// ============================================================================
struct CudaStreamDeleter {
  void operator()(cudaStream_t stream) const noexcept {
    if (stream != nullptr) {
      cudaStreamDestroy(stream);
    }
  }
};

struct CudaDeviceMemDeleter {
  void operator()(void* ptr) const noexcept {
    if (ptr != nullptr) {
      cudaFree(ptr);
    }
  }
};

template <typename T>
struct TrtDeleter {
  void operator()(T* obj) const noexcept {
    delete obj;  // TensorRT 10+ uses standard virtual destructors
  }
};

class TrtLogger final : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char* msg) noexcept override {
    if (severity <= Severity::kWARNING) {
      RCLCPP_WARN(rclcpp::get_logger("TensorRT"), "[NIST-DETECT] %s", msg);
    }
  }
};
#endif

// ============================================================================
// Data Structures: Lock-Guarded Ring Buffer for IMU & Time-Stamped Poses
// ============================================================================
struct ImuMeasurement {
  double timestamp_sec{0.0};
  Eigen::Vector3d angular_velocity{Eigen::Vector3d::Zero()};
  Eigen::Vector3d linear_acceleration{Eigen::Vector3d::Zero()};
};

struct SemanticVoxel {
  float tsdf_value{1.0F};
  float weight{0.0F};
  uint16_t semantic_label{0};
  float label_probability{0.0F};
};

// Spatial Hash for 3D Voxel Grid: O(1) average lookup
struct VoxelIndexHash {
  std::size_t operator()(const Eigen::Vector3i& idx) const noexcept {
    return static_cast<std::size_t>((idx.x() * 73856093) ^
                                    (idx.y() * 19349663) ^
                                    (idx.z() * 83492791));
  }
};

// ============================================================================
// Ceres Cost Functor: Tightly-Coupled Point-to-Plane ICP with Online Temporal
// Offset (td) Compensation.
// Paper Reference: Continuous-time / kinematic deskewing (FAST-LIVO2, T-RO 24)
// ============================================================================
struct TimeCompensatedPointToPlaneFactor {
  TimeCompensatedPointToPlaneFactor(Eigen::Vector3d raw_pt,
                                    Eigen::Vector3d plane_unit_norm,
                                    double plane_d,
                                    Eigen::Vector3d omega_imu,
                                    Eigen::Vector3d vel_imu,
                                    double pt_rel_dt)
      : p_lidar_(std::move(raw_pt)),
        n_world_(std::move(plane_unit_norm)),
        d_world_(plane_d),
        omega_(std::move(omega_imu)),
        vel_(std::move(vel_imu)),
        dt_(pt_rel_dt) {}

  template <typename T>
  bool operator()(const T* const q_w_b,      // Quaternion [x, y, z, w]
                  const T* const t_w_b,      // Translation [x, y, z]
                  const T* const td_offset,  // Online time offset [1]
                  T* residuals) const {
    const T total_dt = T(dt_) + td_offset[0];

    Eigen::Map<const Eigen::Quaternion<T>> q_wb(q_w_b);
    Eigen::Map<const Eigen::Matrix<T, 3, 1>> t_wb(t_w_b);

    const Eigen::Matrix<T, 3, 1> p_l = p_lidar_.cast<T>();
    const Eigen::Matrix<T, 3, 1> ang_vel = omega_.cast<T>();
    const Eigen::Matrix<T, 3, 1> lin_vel = vel_.cast<T>();

    // Deskew point in body frame: p_deskewed = (I + [omega]_x * dt) * p_l + v * dt
    const Eigen::Matrix<T, 3, 1> p_body =
        p_l + ang_vel.cross(p_l) * total_dt + lin_vel * total_dt;

    // Transform to world frame
    const Eigen::Matrix<T, 3, 1> p_world = q_wb * p_body + t_wb;

    // Point-to-plane signed distance residual
    residuals[0] = n_world_.cast<T>().dot(p_world) + T(d_world_);
    return true;
  }

  static ceres::CostFunction* Create(const Eigen::Vector3d& raw_pt,
                                     const Eigen::Vector3d& plane_unit_norm,
                                     double plane_d,
                                     const Eigen::Vector3d& omega_imu,
                                     const Eigen::Vector3d& vel_imu,
                                     double pt_rel_dt) {
    return new ceres::AutoDiffCostFunction<TimeCompensatedPointToPlaneFactor,
                                           1, 4, 3, 1>(
        new TimeCompensatedPointToPlaneFactor(
            raw_pt, plane_unit_norm, plane_d, omega_imu, vel_imu, pt_rel_dt));
  }

 private:
  Eigen::Vector3d p_lidar_;
  Eigen::Vector3d n_world_;
  double d_world_;
  Eigen::Vector3d omega_;
  Eigen::Vector3d vel_;
  double dt_;
};

// ============================================================================
// Main ROS 2 Node: Visual-Inertial-LiDAR Semantic SLAM
// Uses Intra-Process Zero-Copy & Loaned Messages
// ============================================================================
class VilSemanticSlamNode final : public rclcpp::Node {
 public:
  explicit VilSemanticSlamNode(const rclcpp::NodeOptions& options)
      : rclcpp::Node("vil_semantic_slam_node",
                     rclcpp::NodeOptions(options).use_intra_process_comms(true)) {
    const std::string engine_path =
        this->declare_parameter<std::string>("trt_engine_path", "");
    voxel_size_ = this->declare_parameter<double>("voxel_leaf_size", 0.15);

#ifdef ENABLE_TENSORRT
    if (!engine_path.empty()) {
      InitializeTensorRT(engine_path);
    }
#else
    (void)engine_path;
#endif

    kdtree_map_ = std::make_unique<pcl::KdTreeFLANN<pcl::PointXYZINormal>>();
    local_map_cloud_ = pcl::make_shared<pcl::PointCloud<pcl::PointXYZINormal>>();

    auto qos = rclcpp::SensorDataQoS();

    imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
        "/imu/data", qos,
        [this](sensor_msgs::msg::Imu::UniquePtr msg) {
          this->OnImuReceived(std::move(msg));
        });

    rgbd_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
        "/camera/color/image_raw", qos,
        [this](sensor_msgs::msg::Image::UniquePtr msg) {
          this->OnImageReceived(std::move(msg));
        });

    lidar_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        "/lidar/points", qos,
        [this](sensor_msgs::msg::PointCloud2::UniquePtr msg) {
          this->OnLidarReceived(std::move(msg));
        });

    semantic_cloud_pub_ =
        this->create_publisher<sensor_msgs::msg::PointCloud2>(
            "/slam/semantic_map", rclcpp::QoS(10));

    RCLCPP_INFO(this->get_logger(),
                "[NIST-GOVERN] VIL-Semantic SLAM initialized with Intra-Process Zero-Copy.");
  }

  ~VilSemanticSlamNode() override = default;

  VilSemanticSlamNode(const VilSemanticSlamNode&) = delete;
  VilSemanticSlamNode& operator=(const VilSemanticSlamNode&) = delete;

 private:
  void OnImuReceived(sensor_msgs::msg::Imu::UniquePtr msg) {
    const double ts = rclcpp::Time(msg->header.stamp).seconds();

    const Eigen::Vector3d gyr(msg->angular_velocity.x,
                              msg->angular_velocity.y,
                              msg->angular_velocity.z);
    const Eigen::Vector3d acc(msg->linear_acceleration.x,
                              msg->linear_acceleration.y,
                              msg->linear_acceleration.z);
    if (!gyr.allFinite() || !acc.allFinite() || acc.norm() > 50.0) {
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                           "[NIST-DETECT] Anomalous IMU reading rejected.");
      return;
    }

    std::unique_lock<std::shared_mutex> lock(imu_mutex_);
    if (!imu_buffer_.empty() && ts <= imu_buffer_.back().timestamp_sec) {
      RCLCPP_WARN(this->get_logger(),
                  "[NIST-DETECT] Non-monotonic IMU timestamp detected!");
      return;
    }
    imu_buffer_.push_back(ImuMeasurement{ts, gyr, acc});
    if (imu_buffer_.size() > kMaxImuBufferSize) {
      imu_buffer_.erase(imu_buffer_.begin());
    }
  }

  void OnImageReceived(sensor_msgs::msg::Image::UniquePtr msg) {
    if (msg->width == 0 || msg->height == 0 || msg->data.empty()) {
      return;
    }

    const cv::Mat raw_img(static_cast<int>(msg->height),
                          static_cast<int>(msg->width), CV_8UC3,
                          const_cast<uint8_t*>(msg->data.data()));

    cv::Mat mask(raw_img.rows, raw_img.cols, CV_8UC1, cv::Scalar(0));

#ifdef ENABLE_TENSORRT
    if (trt_context_ && cuda_stream_) {
      const std::size_t img_bytes = raw_img.total() * raw_img.elemSize();
      if (img_bytes <= allocated_input_bytes_) {
        cudaMemcpyAsync(d_input_.get(), raw_img.data, img_bytes,
                        cudaMemcpyHostToDevice, cuda_stream_.get());
        trt_context_->enqueueV3(cuda_stream_.get());
        cudaMemcpyAsync(mask.data, d_output_.get(),
                        static_cast<std::size_t>(mask.total()),
                        cudaMemcpyDeviceToHost, cuda_stream_.get());
        cudaStreamSynchronize(cuda_stream_.get());
      }
    }
#endif

    {
      std::unique_lock<std::shared_mutex> lock(semantic_mutex_);
      latest_semantic_mask_ = std::move(mask);
    }
  }

  void OnLidarReceived(sensor_msgs::msg::PointCloud2::UniquePtr msg) {
    auto raw_cloud = pcl::make_shared<pcl::PointCloud<pcl::PointXYZINormal>>();
    pcl::fromROSMsg(*msg, *raw_cloud);
    if (raw_cloud->empty()) {
      return;
    }

    auto downsampled = pcl::make_shared<pcl::PointCloud<pcl::PointXYZINormal>>();
    pcl::VoxelGrid<pcl::PointXYZINormal> sor;
    sor.setInputCloud(raw_cloud);
    const auto leaf = static_cast<float>(voxel_size_);
    sor.setLeafSize(leaf, leaf, leaf);
    sor.filter(*downsampled);

    Eigen::Vector3d latest_omega = Eigen::Vector3d::Zero();
    {
      std::shared_lock<std::shared_mutex> lock(imu_mutex_);
      if (!imu_buffer_.empty()) {
        latest_omega = imu_buffer_.back().angular_velocity;
      }
    }

    cv::Mat current_mask;
    {
      std::shared_lock<std::shared_mutex> lock(semantic_mutex_);
      if (!latest_semantic_mask_.empty()) {
        current_mask = latest_semantic_mask_;
      }
    }

    auto static_cloud = pcl::make_shared<pcl::PointCloud<pcl::PointXYZINormal>>();
    static_cloud->reserve(downsampled->size());

    for (const auto& pt : downsampled->points) {
      if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z)) {
        continue;
      }
      const uint16_t sem_label = QuerySemanticLabelForPoint(pt, current_mask);
      if (sem_label == 1 || sem_label == 2) {
        continue;
      }
      pcl::PointXYZINormal p_clean = pt;
      p_clean.curvature = static_cast<float>(sem_label);
      static_cloud->push_back(p_clean);
    }

    if (local_map_cloud_->size() > 100 && static_cloud->size() > 30) {
      OptimizePoseAndTimeOffset(static_cloud, latest_omega);
    }

    UpdateSemanticVoxelMap(static_cloud);
    PublishZeroCopySemanticMap(msg->header.stamp);
  }

  void OptimizePoseAndTimeOffset(
      const pcl::PointCloud<pcl::PointXYZINormal>::ConstPtr& cloud,
      const Eigen::Vector3d& omega_imu) {
    ceres::Problem problem;
    ceres::Manifold* quat_manifold = new ceres::EigenQuaternionManifold();
    problem.AddParameterBlock(q_w_b_.coeffs().data(), 4, quat_manifold);
    problem.AddParameterBlock(t_w_b_.data(), 3);
    problem.AddParameterBlock(&td_lidar_imu_, 1);

    problem.SetParameterLowerBound(&td_lidar_imu_, 0, -0.05);
    problem.SetParameterUpperBound(&td_lidar_imu_, 0, 0.05);

    std::vector<int> nn_indices(5);
    std::vector<float> nn_sq_dists(5);

    for (const auto& pt : cloud->points) {
      const Eigen::Vector3d p_body(pt.x, pt.y, pt.z);
      const Eigen::Vector3d p_world = q_w_b_ * p_body + t_w_b_;

      pcl::PointXYZINormal search_pt;
      search_pt.x = static_cast<float>(p_world.x());
      search_pt.y = static_cast<float>(p_world.y());
      search_pt.z = static_cast<float>(p_world.z());

      if (kdtree_map_->nearestKSearch(search_pt, 5, nn_indices, nn_sq_dists) == 5 &&
          nn_sq_dists[4] < 1.0F) {
        Eigen::Matrix<double, 5, 3> A;
        Eigen::Matrix<double, 5, 1> b = Eigen::Matrix<double, 5, 1>::Constant(-1.0);
        for (int i = 0; i < 5; ++i) {
          const auto& nbr = local_map_cloud_->points[static_cast<std::size_t>(nn_indices[i])];
          A.row(i) << nbr.x, nbr.y, nbr.z;
        }
        const Eigen::Vector3d norm_raw = A.colPivHouseholderQr().solve(b);
        const double norm_len = norm_raw.norm();
        if (norm_len > 1e-4) {
          const Eigen::Vector3d unit_norm = norm_raw / norm_len;
          const double plane_d = 1.0 / norm_len;

          ceres::LossFunction* loss = new ceres::HuberLoss(0.1);
          ceres::CostFunction* cost = TimeCompensatedPointToPlaneFactor::Create(
              p_body, unit_norm, plane_d, omega_imu, v_w_b_, 0.0);
          problem.AddResidualBlock(
              cost, loss, q_w_b_.coeffs().data(), t_w_b_.data(), &td_lidar_imu_);
        }
      }
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.max_num_iterations = 6;
    options.num_threads = 4;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
  }

  uint16_t QuerySemanticLabelForPoint(const pcl::PointXYZINormal& pt,
                                      const cv::Mat& mask) const noexcept {
    if (mask.empty() || pt.x <= 0.1F) {
      return 0;
    }
    constexpr float fx = 525.0F, fy = 525.0F, cx = 319.5F, cy = 239.5F;
    const int u = static_cast<int>((fx * (-pt.y) / pt.x) + cx);
    const int v = static_cast<int>((fy * (-pt.z) / pt.x) + cy);
    if (u >= 0 && u < mask.cols && v >= 0 && v < mask.rows) {
      return static_cast<uint16_t>(mask.at<uint8_t>(v, u));
    }
    return 0;
  }

  void UpdateSemanticVoxelMap(
      const pcl::PointCloud<pcl::PointXYZINormal>::ConstPtr& static_cloud) {
    for (const auto& pt : static_cloud->points) {
      const Eigen::Vector3d p_w =
          q_w_b_ * Eigen::Vector3d(pt.x, pt.y, pt.z) + t_w_b_;
      const Eigen::Vector3i v_idx(
          static_cast<int>(std::floor(p_w.x() / voxel_size_)),
          static_cast<int>(std::floor(p_w.y() / voxel_size_)),
          static_cast<int>(std::floor(p_w.z() / voxel_size_)));

      auto& voxel = semantic_voxel_grid_[v_idx];
      const uint16_t obs_label = static_cast<uint16_t>(pt.curvature);
      if (voxel.semantic_label == obs_label) {
        voxel.label_probability = std::min(1.0F, voxel.label_probability + 0.2F);
      } else if (voxel.label_probability <= 0.3F) {
        voxel.semantic_label = obs_label;
        voxel.label_probability = 0.6F;
      } else {
        voxel.label_probability -= 0.1F;
      }
      voxel.weight = std::min(100.0F, voxel.weight + 1.0F);
    }

    local_map_cloud_->clear();
    local_map_cloud_->reserve(semantic_voxel_grid_.size());
    for (const auto& [idx, vox] : semantic_voxel_grid_) {
      pcl::PointXYZINormal p;
      p.x = static_cast<float>((idx.x() + 0.5) * voxel_size_);
      p.y = static_cast<float>((idx.y() + 0.5) * voxel_size_);
      p.z = static_cast<float>((idx.z() + 0.5) * voxel_size_);
      p.intensity = static_cast<float>(vox.semantic_label);
      local_map_cloud_->push_back(p);
    }
    if (!local_map_cloud_->empty()) {
      kdtree_map_->setInputCloud(local_map_cloud_);
    }
  }

  void PublishZeroCopySemanticMap(const builtin_interfaces::msg::Time& stamp) {
    auto out_msg = std::make_unique<sensor_msgs::msg::PointCloud2>();
    pcl::toROSMsg(*local_map_cloud_, *out_msg);
    out_msg->header.stamp = stamp;
    out_msg->header.frame_id = "map";
    semantic_cloud_pub_->publish(std::move(out_msg));
  }

#ifdef ENABLE_TENSORRT
  void InitializeTensorRT(const std::string& engine_path) {
    std::ifstream file(engine_path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
      RCLCPP_ERROR(this->get_logger(),
                   "[NIST-PROTECT] Failed to open TensorRT engine: %s",
                   engine_path.c_str());
      return;
    }
    const std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> buffer(static_cast<std::size_t>(size));
    if (!file.read(buffer.data(), size)) {
      return;
    }

    trt_runtime_.reset(nvinfer1::createInferRuntime(trt_logger_));
    if (!trt_runtime_) return;

    trt_engine_.reset(
        trt_runtime_->deserializeCudaEngine(buffer.data(), buffer.size()));
    if (!trt_engine_) return;

    trt_context_.reset(trt_engine_->createExecutionContext());

    cudaStream_t raw_stream = nullptr;
    if (cudaStreamCreate(&raw_stream) == cudaSuccess) {
      cuda_stream_.reset(raw_stream);
    }

    allocated_input_bytes_ = 640ULL * 480ULL * 3ULL;
    void* raw_in = nullptr;
    void* raw_out = nullptr;
    if (cudaMalloc(&raw_in, allocated_input_bytes_) == cudaSuccess) {
      d_input_.reset(raw_in);
    }
    if (cudaMalloc(&raw_out, 640ULL * 480ULL) == cudaSuccess) {
      d_output_.reset(raw_out);
    }
  }
#endif

  static constexpr std::size_t kMaxImuBufferSize = 2000;
  double voxel_size_{0.15};

  std::shared_mutex imu_mutex_;
  std::vector<ImuMeasurement> imu_buffer_;

  std::shared_mutex semantic_mutex_;
  cv::Mat latest_semantic_mask_;

  Eigen::Quaterniond q_w_b_{Eigen::Quaterniond::Identity()};
  Eigen::Vector3d t_w_b_{Eigen::Vector3d::Zero()};
  Eigen::Vector3d v_w_b_{Eigen::Vector3d::Zero()};
  double td_lidar_imu_{0.0};

  std::unordered_map<Eigen::Vector3i, SemanticVoxel, VoxelIndexHash>
      semantic_voxel_grid_;
  pcl::PointCloud<pcl::PointXYZINormal>::Ptr local_map_cloud_;
  std::unique_ptr<pcl::KdTreeFLANN<pcl::PointXYZINormal>> kdtree_map_;

#ifdef ENABLE_TENSORRT
  TrtLogger trt_logger_;
  std::unique_ptr<nvinfer1::IRuntime, TrtDeleter<nvinfer1::IRuntime>> trt_runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine, TrtDeleter<nvinfer1::ICudaEngine>> trt_engine_;
  std::unique_ptr<nvinfer1::IExecutionContext, TrtDeleter<nvinfer1::IExecutionContext>> trt_context_;
  std::unique_ptr<CUstream_st, CudaStreamDeleter> cuda_stream_;
  std::unique_ptr<void, CudaDeviceMemDeleter> d_input_;
  std::unique_ptr<void, CudaDeviceMemDeleter> d_output_;
  std::size_t allocated_input_bytes_{0};
#endif

  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr rgbd_sub_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr lidar_sub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr semantic_cloud_pub_;
};

}  // namespace vil_slam
