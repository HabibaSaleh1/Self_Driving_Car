# Self-Driving Car Simulation (YOLOv8 + OpenCV)

Real-time self-driving car simulation built using computer vision and deep learning. The system performs lane detection, object detection, and driving decision-making in a simulated highway environment.

---

## Features

- Real-time lane detection using Canny Edge Detection + Hough Transform
- YOLOv8-based object detection (cars, trucks, buses, pedestrians, etc.)
- Ego-lane filtering for relevant object tracking
- Proximity estimation (FAR / NEAR / CLOSE)
- Driving decision engine:
  - Go Straight
  - Turn Left / Right
  - Slow Down
  - Stop (Pedestrian / Vehicle ahead)
- Decision smoothing using buffer logic
- Real-time HUD overlay for visualization
- Optimized inference (runs YOLO every N frames)

---

## Tech Stack

- Python
- OpenCV
- NumPy
- YOLOv8 (Ultralytics)
- Computer Vision techniques (Canny, Hough Transform)


## Project Structure


.
├── main.py
├── yolov8n.pt
├── Highway.mp4
├── README.md



## Installation

```bash
git clone :(https://github.com/HabibaSaleh1/Self_Driving_Car.git)
cd Self_Driving_Car

install requirements

pip install opencv-python numpy ultralytics
Run the Project
python main.py

Press Q to exit the simulation.
