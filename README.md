# Parking Space Detection System

## Overview

This project implements a computer vision-based system for detecting and monitoring parking space availability. The system supports two detection methods:

1. **Traditional CV Method**: Using OpenCV image processing techniques (thresholding, contours)
2. **ML-Based Method**: Using YOLOv8 deep learning model for vehicle detection

## Features

- **Interactive Parking Space Marking**: Click and drag to define parking spaces
- **Real-time Space Monitoring**: Track and display free/occupied spaces
- **Dual Detection Methods**:
  - OpenCV-based processing with adaptive parameters
  - YOLOv8-based vehicle detection with IoU calculation
- **Persistence**: Parking space positions are saved between sessions
- **Adjustable Parameters**: Fine-tune detection with trackbars

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/varun5115/Cars-Parking-Detector.git
   cd Cars-Parking-Detector
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the YOLOv8 model (if using ML-based detection):
   ```bash
   # The model will be downloaded automatically when first using the ML method
   # or download manually from https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
   ```

4. Place a video file named `car_parking.mp4` in the project directory or update the `VIDEO_PATH` in `config.py`.

## Usage

### Running the Application

Run with traditional OpenCV-based detection (default):
```bash
python main.py
```

Run with YOLO-based ML detection:
```bash
python main.py --method ml
```

### Workflow

1. **First Run - Marking Mode**:
   - The application starts in marking mode if no parking spaces are defined
   - Click and drag to define rectangular parking spaces
   - Press 'q' or ESC to switch to detection mode

2. **Detection Mode**:
   - The system will analyze the video frame-by-frame
   - Green rectangles indicate free spaces, red rectangles indicate occupied spaces
   - The count of free/total spaces is displayed in the top left
   - Adjust parameters with the trackbars (for OpenCV method)

### Parameter Tuning

The OpenCV method uses three parameters that can be adjusted in real-time:
- **Val1**: Block size for adaptive thresholding
- **Val2**: Constant subtracted from the mean
- **Val3**: Kernel size for median blur

## Project Structure

```
parking_space/
├── main.py              # Main application entry point
├── parking_logic.py     # Core detection and space management functions
├── video_processing.py  # Video handling and frame preprocessing
├── config.py            # Configuration parameters
├── CarParkPos           # Binary file storing parking space positions
├── yolov8n.pt          # YOLOv8 model file (downloaded when needed)
├── requirements.txt     # Project dependencies
└── README.md            # This documentation
```

## Configuration

You can modify the following parameters in `config.py`:

```python
# File paths
VIDEO_PATH = 'car_parking.mp4'  # Path to input video
PARKING_FILE = 'CarParkPos'     # File to store parking spaces

# Detection parameters
PARKING_THRESHOLD = 500         # Pixel count threshold for occupancy

# UI Parameters
WINDOW_NAME = "Vals"            # Name of parameter window
WINDOW_SIZE = (640, 240)        # Size of parameter window

# Pre-processing parameters
GAUSSIAN_BLUR_KERNEL = (3, 3)   # Kernel size for Gaussian blur
GAUSSIAN_BLUR_SIGMA = 1         # Sigma for Gaussian blur
DILATION_KERNEL = (3, 3)        # Kernel size for dilation
DILATION_ITERATIONS = 1         # Number of dilation iterations
```

## Technical Details

### OpenCV-based Detection

The traditional method uses these steps:
1. Convert frame to grayscale
2. Apply Gaussian blur
3. Apply adaptive thresholding
4. Count non-zero pixels in each parking space
5. Determine occupancy based on pixel count threshold

### YOLOv8-based Detection

The ML-based method:
1. Detects vehicles in the entire frame using YOLOv8
2. For each parking space, calculates overlap with detected vehicles
3. Determines occupancy based on Intersection over Union (IoU)

## Requirements

See [requirements.txt](requirements.txt) for the complete list of dependencies.

## License

[MIT License](LICENSE)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
