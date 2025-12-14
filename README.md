# Pose Estimator

A simple pose estimation tool that detects human body poses in video and draws a skeleton overlay using MediaPipe and OpenCV.

## Features

- Detects 33 body landmarks (joints) per person
- Draws skeleton overlay with joints and bone connections
- Face landmarks excluded by default (optional)
- Processes video files with progress reporting
- Optional live preview window

## Requirements

- Python 3.8+
- OpenCV
- MediaPipe

## Setup

### Windows (Native)

1. **Install Python** (if not already installed)
   - Download from https://www.python.org/downloads/
   - During installation, check "Add Python to PATH"

2. **Create a virtual environment** (recommended)
   ```cmd
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install dependencies**
   ```cmd
   pip install opencv-python mediapipe
   ```

4. **Verify installation**
   ```cmd
   python -c "import cv2; import mediapipe; print('Ready!')"
   ```

### Windows Subsystem for Linux (WSL)

1. **Ensure WSL is installed**
   ```powershell
   # In PowerShell (as Administrator)
   wsl --install
   ```

2. **Open WSL terminal** and install Python
   ```bash
   sudo apt update
   sudo apt install python3 python3-pip python3-venv
   ```

3. **Create a virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

4. **Install dependencies**
   ```bash
   pip install opencv-python mediapipe
   ```

5. **Note on video preview in WSL**
   - The `--preview` flag requires a display server
   - If using WSL2 with WSLg (Windows 11), it should work automatically
   - Otherwise, skip `--preview` and just generate output files

### macOS / Linux

1. **Create a virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install dependencies**
   ```bash
   pip install opencv-python mediapipe
   ```

## Usage

### Basic Usage

```bash
# Process a video and save output
python pose_estimator.py input.mp4 -o output.mp4

# Windows example with full paths
python pose_estimator.py "C:\Users\Name\Videos\workout.mp4" -o "C:\Users\Name\Videos\output.mp4"

# WSL example (accessing Windows files)
python pose_estimator.py /mnt/c/Users/Name/Videos/workout.mp4 -o /mnt/c/Users/Name/Videos/output.mp4
```

### Options

```bash
# Show live preview while processing (press 'q' to quit)
python pose_estimator.py input.mp4 -o output.mp4 --preview

# Include face landmarks (eyes, ears, nose, mouth)
python pose_estimator.py input.mp4 -o output.mp4 --with-face

# Adjust detection confidence (higher = fewer false positives)
python pose_estimator.py input.mp4 -o output.mp4 --detection-confidence 0.7

# Adjust tracking confidence (higher = more stable between frames)
python pose_estimator.py input.mp4 -o output.mp4 --tracking-confidence 0.7
```

### Full Options List

```
usage: pose_estimator.py [-h] [-o OUTPUT] [--preview]
                         [--detection-confidence DETECTION_CONFIDENCE]
                         [--tracking-confidence TRACKING_CONFIDENCE]
                         [--with-face]
                         input

positional arguments:
  input                 Input video file path

options:
  -h, --help            Show help message
  -o, --output          Output video file path
  --preview             Show live preview window
  --detection-confidence
                        Minimum detection confidence 0.0-1.0 (default: 0.5)
  --tracking-confidence
                        Minimum tracking confidence 0.0-1.0 (default: 0.5)
  --with-face           Include face landmarks
```

## Example Workflow

```bash
# 1. Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/macOS/WSL:
source venv/bin/activate

# 2. Process a workout video
python pose_estimator.py workout.mp4 -o workout_pose.mp4

# 3. Check the output
# The output video will have skeleton overlay drawn on detected persons
```

## Troubleshooting

### "ModuleNotFoundError: No module named 'cv2'"
```bash
pip install opencv-python
```

### "ModuleNotFoundError: No module named 'mediapipe'"
```bash
pip install mediapipe
```

### Video output is empty or corrupted
- Ensure the input video codec is supported
- Try converting input to MP4 with H.264 codec first

### Preview window doesn't appear (WSL)
- WSL requires WSLg or an X server for GUI
- Use without `--preview` flag and view the output file instead

### Slow processing
- Lower resolution videos process faster
- Consider reducing `model_complexity` in the code (0=lite, 1=full, 2=heavy)

## How It Works

See [HOW_IT_WORKS.md](HOW_IT_WORKS.md) for a technical explanation of the pose estimation pipeline.

## License

MIT
