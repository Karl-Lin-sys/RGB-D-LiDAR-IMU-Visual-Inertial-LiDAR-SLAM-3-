# 🍁 Real-Time Visual-Inertial-LiDAR (RGB-D + LiDAR + IMU) SLAM & 3D Semantic Mapping

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Standard](https://img.shields.io/badge/C%2B%2B-20-blue.svg)](https://en.cppreference.com/w/cpp/20)
[![ROS 2](https://img.shields.io/badge/ROS_2-Humble_%7C_Jazzy-22314E.svg)](https://docs.ros.org/)
[![NIST CSF](https://img.shields.io/badge/NIST_CSF-2.0_Compliant-success.svg)](https://www.nist.gov/cyberframework)

## 🦫 Overview (Canadian Toddler Edition, eh!)

Hey there, little buddy! Grab a Timbit and let's see how our robot beaver finds its way through a snowy forest!
Our robot wears three kinds of magic goggles:
1. **RGB-D Camera (Colour & Depth Goggles):** Sees pretty colours and how far away the maple trees are!
2. **LiDAR (Laser Bouncer):** Shoots invisible light beams (*pew, pew!*) to measure every bump in the snow!
3. **IMU (Tummy Balance Sensor):** Feels when the robot tilts, spins, or zooms fast on a toboggan!

By combining all three using **Modern C++20**, **ROS 2 Intra-Process Zero-Copy**, **Ceres Solver ($SE(3)$ Manifold + Online Temporal Calibration)**, **PCL**, **Eigen**, **OpenCV**, and **TensorRT**, our robot builds a 3D map and labels obstacles in real time without leaking a single drop of memory!

---

## 🛡️ NIST Cybersecurity Framework (CSF 2.0) Architecture

| NIST CSF 2.0 Function | Toddler Explanation (Canadian Style, eh!) | Technical Implementation in C++20 |
| :--- | :--- | :--- |
| **GOVERN (GV)** | Making the house rules before we play! | Modern C++20 Core Guidelines, strict RAII (`std::unique_ptr`), and Taiwan R.O.C. Open-Source Legal Compliance (`Apache-2.0` SPDX headers). |
| **IDENTIFY (ID)** | Knowing which toys can break if we step on them. | Mitigating Time Synchronization Drift, Extrinsic Calibration Drift on $SE(3)$, and GPU buffer overflows. |
| **PROTECT (PR)** | Wearing a hockey helmet and knee pads! | Custom RAII deleters (`CudaStreamDeleter`, `CudaDeviceMemDeleter`, `TrtDeleter`), parameter bounds on time offsets, and `.gitignore` exclusions for weights/recordings. |
| **DETECT (DT)** | Smelling burnt toast right away! | Monotonic IMU timestamp checks, norm bounds for spoofing/anomaly rejection, and dynamic semantic object masking. |
| **RESPOND (RS)** | Dropping the puck and skating back on defence! | Huber loss kernels and degeneracy-aware localizability handling (inspired by *X-ICP*, IEEE T-RO 2024). |
| **RECOVER (RC)** | Picking yourself up off the ice and cheering! | Automatic resource reclamation via RAII and sliding-window voxel hash map reconstruction. |

---

## 📚 Peer-Reviewed Architecture Comparison (2023–2026)

* **FAST-LIVO2 / R3LIVE++** (*Zheng et al., IEEE T-RO 2024/2025; Lin & Zhang, IEEE T-RO 2023*): Direct tightly-coupled ESIKF fusing sparse visual patches and LiDAR points in a unified voxel/ikd-Tree map with online temporal offset estimation.
* **X-ICP** (*Tuna et al., IEEE T-RO 2024*): Localizability-aware LiDAR registration analyzing Hessian eigenvalues to prevent divergence in geometrically degenerate tunnels/corridors.
* **Hydra / Khronos** (*Hughes et al., IJRR 2024; Schmid et al., RSS 2024*): Real-time 3D metric-semantic mapping and spatio-temporal dynamic object filtering using Bayesian voxel updates.

---

## ⏱️ Algorithmic Complexity & Real-Time Latency

* **$N$**: Raw LiDAR points per scan (~50,000)
* **$M$**: Voxel-downsampled static points (~2,000)
* **$V$**: Active 3D semantic voxels in hash map

| Component | Time Complexity | Space Complexity | Real-Time Target |
| :--- | :--- | :--- | :--- |
| `OnImuReceived` | $\mathcal{O}(1)$ | $\mathcal{O}(B)$ ($B=2000$) | `< 0.05 ms` |
| `OnImageReceived` (TensorRT) | $\mathcal{O}(W \cdot H)$ | $\mathcal{O}(W \cdot H)$ | `< 6.5 ms` |
| `VoxelGrid` Downsampling | $\mathcal{O}(N)$ | $\mathcal{O}(M)$ | `~ 1.5 ms` |
| `OptimizePoseAndTimeOffset` | $\mathcal{O}(I \cdot M \log V)$ | $\mathcal{O}(M)$ | `< 12.0 ms` |
| `UpdateSemanticVoxelMap` | $\mathcal{O}(M)$ average | $\mathcal{O}(V)$ | `~ 2.0 ms` |

---

## 🇹🇼 Taiwan (R.O.C.) Legal Compliance Notice

This repository is structured to comply with Taiwan (R.O.C.) laws:
1. **Copyright Act (`著作權法`):** Licensed explicitly under the **Apache License 2.0** (`SPDX-License-Identifier: Apache-2.0`).
2. **Trade Secrets Act (`營業秘密法`) & National Security Act (`國家安全法`):** No proprietary weights (`.onnx`, `.engine`) or private factory extrinsics are included.
3. **Personal Data Protection Act (`個人資料保護法` - PDPA):** No raw sensor recordings (`.bag`, `.db3`, `.mcap`) containing identifiable faces or license plates are tracked.
