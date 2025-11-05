# VisionAid: Real-Time Obstacle Detection & Audio Feedback for Visually Impaired Navigation

VisionAid is a software-only, real-time obstacle detection and audio feedback system designed to assist visually impaired individuals in navigating low-light environments. 
The app uses classical computer vision techniques enhanced with image processing (CLAHE, gamma correction, denoising) and provides real-time audio feedback for detected obstacles.

This repository contains the full Python implementation of the pipeline, including image enhancement, obstacle detection, distance estimation, and audio feedback modules.

---

## Features

- **Real-Time Obstacle Detection:** Edge and contour-based detection pipeline enhanced with CLAHE, gamma correction, and denoising for low-light conditions.
- **Distance Estimation:** Supports stereo vision, monocular heuristics, and optional ultrasonic sensor integration.
- **Audio Feedback:** Provides directional audio cues (`left`, `center`, `right`) with proximity bands to alert users about nearby obstacles.
- **Lightweight & Offline:** Runs entirely on standard computing platforms without the need for GPUs or cloud connectivity.
- **Dataset Integration:** Tested on the [DarkFace Dataset](https://www.kaggle.com/datasets/soumikrakshit/dark-face-dataset) for low-light obstacle scenarios.

---

## Dataset

The system was evaluated using the **DarkFace Dataset**, which includes:

- 6,000 real-world low-light images (labeled for human faces) for training/validation.
- 9,000 additional unlabeled low-light images.
- 789 paired low-light/normal-light images in controllable lighting.
- A hold-out test set of 4,000 low-light images with bounding box annotations.
---

## Installation

1. **Clone the repository**

```bash
git clone https://github.com/adya0217/VisionAid.git
cd VisionAid
```
2. **Install Requiremnets**
```bash
pip install -r requirements.txt
```

**Run the app**
```bash
python main.py
```

Once running, you can:\
Use a connected webcam for live obstacle detection.

Receive audio feedback indicating obstacle direction (left, center, right) and proximity (near, far).

Optionally, provide a video file instead of a live stream for testing.

**Features**

Low-Light Obstacle Detection: Detect obstacles in low-light environments using the DarkFace dataset as reference.

Audio Feedback: Provides directional audio cues for safer navigation.

Real-Time Performance: Designed for live camera feeds with minimal latency.

Lightweight & Offline: Runs without cloud dependency or specialized hardware.


**Conatct Details**
adya.gangwal@gmail.com
nehajrao26@gmail.com
