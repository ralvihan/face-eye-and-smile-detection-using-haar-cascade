# Real-Time Face, Eye, and Smile Detection

A real-time facial feature detection project built with OpenCV and Haar Cascade Classifiers. It detects faces, eyes, and smiles from a webcam feed, either separately or combined, and includes a script that records detection counts over 15 seconds and plots them as a chart.

## Project Structure

| File | Description |
|---|---|
| `faceDetection.py` | Face detection |
| `faceEyeDetection.py` | Face and eye detection |
| `faceSmileDetection.py` | Face and smile detection |
| `faceEyeSmileDetection.py` | Face, eye, and smile detection combined |
| `grafik.py` | Detection count chart over 15 seconds |

## Tech Stack

Python, OpenCV, NumPy, Matplotlib

## Getting Started

```bash
pip install opencv-python numpy matplotlib
python faceEyeSmileDetection.py
```

## Result

![Detection count chart](figure.png)
