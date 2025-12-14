# Pose Estimation: How It Works

This document explains the basics of how `pose_estimator.py` detects and draws human body poses on video.

## Overview

The system uses **MediaPipe Pose**, a machine learning model from Google that detects human body positions in images/video. It identifies 33 "landmarks" (key points) on the body and returns their coordinates, which we then use to draw a skeleton overlay.

## The Pipeline

```
Video Frame (BGR) → Convert to RGB → MediaPipe Detection → Draw Skeleton → Output Frame
```

### Step 1: Read Video Frames

OpenCV (`cv2.VideoCapture`) reads the video file frame-by-frame. Each frame is a numpy array of pixel values in BGR format (Blue-Green-Red).

```python
cap = cv2.VideoCapture("video.mp4")
ret, frame = cap.read()  # frame shape: (height, width, 3)
```

### Step 2: Color Conversion

MediaPipe expects RGB format, but OpenCV uses BGR, so we convert:

```python
rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
```

### Step 3: Pose Detection

MediaPipe's neural network processes the frame and returns landmark positions:

```python
results = pose.process(rgb_frame)
landmarks = results.pose_landmarks  # 33 body points
```

Each landmark has:
- `x`, `y`: Position as a fraction of image width/height (0.0 to 1.0)
- `z`: Depth estimate (relative to hips)
- `visibility`: Confidence that the point is visible (0.0 to 1.0)

### Step 4: Draw the Skeleton

We draw two things:
1. **Joints** - circles at each landmark position
2. **Bones** - lines connecting related landmarks

```python
# Convert normalized coordinates to pixel coordinates
x = int(landmark.x * frame_width)
y = int(landmark.y * frame_height)

# Draw joint
cv2.circle(frame, (x, y), radius=3, color=(0, 255, 0), thickness=2)

# Draw bone (line between two landmarks)
cv2.line(frame, start_point, end_point, color=(255, 0, 0), thickness=2)
```

### Step 5: Write Output

Processed frames are written to a new video file:

```python
writer = cv2.VideoWriter("output.mp4", fourcc, fps, (width, height))
writer.write(annotated_frame)
```

## The 33 Body Landmarks

MediaPipe detects these points:

```
Face (0-10):     nose, eyes, ears, mouth corners
Upper Body:     shoulders (11-12), elbows (13-14), wrists (15-16)
Hands:          pinky/index/thumb (17-22)
Lower Body:     hips (23-24), knees (25-26), ankles (27-28)
Feet:           heels (29-30), foot index (31-32)
```

By default, our script excludes face landmarks (0-10) since they're not useful for body pose analysis.

## Connections (Bones)

MediaPipe defines which landmarks connect to form the skeleton. For example:
- Left shoulder (11) connects to left elbow (13)
- Left elbow (13) connects to left wrist (15)
- Left hip (23) connects to left knee (25)

These connections are pre-defined in `mp.solutions.pose.POSE_CONNECTIONS`.

## Key Parameters

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `min_detection_confidence` | 0.5 | How confident the model must be to detect a new pose |
| `min_tracking_confidence` | 0.5 | How confident to continue tracking between frames |
| `model_complexity` | 1 | Model size: 0=lite, 1=full, 2=heavy (slower but more accurate) |

## Usage Examples

```bash
# Basic usage (face landmarks excluded by default)
python pose_estimator.py input.mp4 -o output.mp4

# Include face landmarks
python pose_estimator.py input.mp4 -o output.mp4 --with-face

# Higher confidence threshold (fewer false detections)
python pose_estimator.py input.mp4 -o output.mp4 --detection-confidence 0.7

# Show live preview while processing
python pose_estimator.py input.mp4 -o output.mp4 --preview
```

## Performance Notes

- Processing speed depends on video resolution and `model_complexity`
- On Apple M1 Pro: ~40-50 fps for 720p video with default settings
- The model runs on CPU via TensorFlow Lite (GPU acceleration available but requires additional setup)

## Further Reading

- [MediaPipe Pose Documentation](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker)
- [OpenCV Video I/O](https://docs.opencv.org/4.x/dd/d43/tutorial_py_video_display.html)
