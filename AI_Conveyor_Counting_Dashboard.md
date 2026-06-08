# README – AI Conveyor Product Counting Dashboard

## Overview
This application is a **Vision AI-based conveyor product counting system** built using **YOLOv5 Segmentation**, **OpenCV**, **PyTorch**, and **Flask**. It detects, tracks, and counts products moving through predefined conveyor regions in real time and provides a live monitoring dashboard through a web browser.

The system is designed for industrial environments where accurate product counting, real-time monitoring, and operational visibility are required.

---

## Key Features

- Real-time product detection using YOLOv5 Segmentation
- Object tracking with unique ID assignment
- Dual conveyor monitoring support
- Region of Interest (ROI) based counting
- Live web dashboard visualization
- Real-time statistics and health monitoring
- Automatic CSV count logging
- Snapshot capture functionality
- Video recording support
- GPU-accelerated inference (CUDA required)
- Event logging and activity tracking

---

## System Architecture

\`\`\`text
Video Source / Camera
          │
          ▼
   YOLOv5 Segmentation
          │
          ▼
   Object Detection
          │
          ▼
      Tracking
          │
          ▼
 ROI-based Counting
          │
 ┌────────┴────────┐
 ▼                 ▼
CSV Logs      Dashboard
                    │
                    ▼
          Real-Time Monitoring
\`\`\`

---

## Functional Modules

### 1. Detection Module
- Loads custom YOLOv5 segmentation model.
- Performs object detection on video frames.
- Generates segmentation masks and bounding boxes.
- Extracts object centroids for tracking.

### 2. Tracking Module
- Assigns unique IDs to detected objects.
- Maintains object state across frames.
- Prevents duplicate counting.
- Removes inactive tracks after timeout.

### 3. Conveyor Counting Module
The system supports:

#### Conveyor 1 (C1)
- Dedicated ROI polygon.
- Independent counting logic.
- Separate statistics and logs.

#### Conveyor 2 (C2)
- Dedicated ROI polygon.
- Independent counting logic.
- Separate statistics and logs.

### 4. Dashboard Module
Provides:

- Live video streaming
- Total product count
- Active object IDs
- FPS monitoring
- Model information
- Device status
- Event logs
- System health metrics

---

## Dashboard APIs

| Endpoint | Description |
|-----------|------------|
| `/` | Dashboard Homepage |
| `/video_feed` | Live Video Stream |
| `/stats` | Detection Statistics |
| `/events` | Event Logs |
| `/health` | System Health |
| `/toggle_inference` | Pause/Resume Inference |
| `/snapshot` | Save Snapshot |
| `/export_csv` | Download Count Log |
| `/shutdown` | Stop Application |

---

## Output Files

### Count Log

\`\`\`text
dashboard_count_log.csv
\`\`\`

Contains:

| Timestamp | Conveyor ID | Tracker ID |
|------------|------------|------------|
| 2026-06-08 10:15:22 | C1 | C1_15 |

---

### Snapshots

\`\`\`text
runs/
 └── snapshots/
      ├── snapshot_20260608_101522.jpg
      └── snapshot_20260608_102130.jpg
\`\`\`

---

### Recorded Videos

\`\`\`text
runs/
 └── videos/
      ├── inference_20260608_101500.mp4
      └── screen_20260608_101500.mp4
\`\`\`

---

## Hardware Requirements

### Minimum

- Intel Core i5 / Ryzen 5
- 16 GB RAM
- NVIDIA GPU (4 GB VRAM)
- 100 GB Storage

### Recommended

- Intel Core i7 / Ryzen 7
- 32 GB RAM
- NVIDIA RTX 3060 / RTX 4060 or higher
- SSD Storage
- CUDA-enabled GPU

---

## Software Requirements

- Windows 10/11
- Python 3.9+
- CUDA Toolkit
- PyTorch with CUDA support
- OpenCV
- Flask
- NumPy
- PIL
- psutil

---

## Installation

### Clone Repository

\`\`\`bash
git clone <repository_url>
cd project
\`\`\`

### Install Dependencies

\`\`\`bash
pip install -r requirements.txt
\`\`\`

---

## Running the Application

### Video File Input

\`\`\`bash
python client_video_dash_board_.py --weights model.pt --source video.mp4 --device 0
\`\`\`

### USB Camera Input

\`\`\`bash
python client_video_dash_board_.py --weights model.pt --source 0 --device 0
\`\`\`

### Screen Recording Mode

\`\`\`bash
python client_video_dash_board_.py --weights model.pt --source video.mp4 --device 0 --save-mode screen --select-screen-region
\`\`\`

---

## ROI Configuration

The application supports custom conveyor ROIs.

Example:

\`\`\`bash
--c1-roi "636,577;971,562;1006,700;649,706"

--c2-roi "1071,80;1090,320;1240,290;1237,80"
\`\`\`

Format:

\`\`\`text
x1,y1;x2,y2;x3,y3;x4,y4
\`\`\`

---

## Monitoring Metrics

The dashboard displays:

- Total product count
- Active tracking IDs
- Detection count
- Inference FPS
- GPU utilization
- CPU utilization
- RAM utilization
- Camera status
- Inference speed (ms)

---

## Industrial Benefits

### Operational Benefits

- Eliminates manual counting
- Improves counting accuracy
- Real-time production visibility
- Reduces human error
- Enables automated reporting
- Supports production analytics

### Business Benefits

- Reduced labor effort
- Improved operational efficiency
- Better production tracking
- Increased scalability
- Data-driven decision making
- Faster issue identification

---

## Expected Outcome

The solution continuously monitors products moving on conveyor belts, accurately detects and tracks each product, and automatically generates count records without manual intervention. The dashboard provides real-time visibility into operations, enabling production teams to monitor throughput, analyze performance, and improve overall operational efficiency.
