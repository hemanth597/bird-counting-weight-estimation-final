bird-counting-weight-estimation/
│
├── app/
│   ├── main.py                     # FastAPI entrypoint
│   ├── video_processor.py          # Detection + tracking
│   ├── weight_estimator.py         # Perspective-aware weight proxy
│   ├── calibration.py              # Camera geometry utils
│
├── data/
│   └── poultry_video.mp4
│
├── models/
│   └── yolov8n.pt
│
├── outputs/
│   ├── annotated_video.mp4
│   └── results.json
│
├── Dockerfile
├── requirements.txt
├── README.md
└── .gitignore
 mathematics
 Weight Index =
    (Bounding Box Area)
  × (Perspective Correction Factor)
  × (Temporal Stability)
def perspective_factor(y_center, frame_height):
    """
    Birds lower in frame are closer to camera.
    Normalize weight by distance approximation.
    """
    norm_y = y_center / frame_height
    return 1.0 + (1 - norm_y) * 0.8
import numpy as np
from app.calibration import perspective_factor

def estimate_weight_proxy(tracks, frame_height):
    """
    Perspective-aware weight index
    """
    weights = {}

    for track_id, observations in tracks.items():
        corrected_areas = []

        for obs in observations:
            area = obs["area"]
            y_center = obs["y_center"]

            p_factor = perspective_factor(y_center, frame_height)
            corrected_areas.append(area * p_factor)

        # Temporal smoothing
        weight_index = np.mean(corrected_areas) / 1200
        weights[int(track_id)] = round(weight_index, 2)

    return weights
self.tracks.setdefault(track_id, []).append({
    "area": area,
    "y_center": (y1 + y2) // 2
})
from fastapi import FastAPI
from app.video_processor import BirdTracker
from app.weight_estimator import estimate_weight_proxy

app = FastAPI()

@app.get("/analyze")
def analyze():
    tracker = BirdTracker()
    tracks, counts, frame_h = tracker.process_video(
        "data/poultry_video.mp4",
        "outputs/annotated_video.mp4"
    )

    weights = estimate_weight_proxy(tracks, frame_h)

    return {
        "unique_birds": len(tracks),
        "sample_counts": counts[:10],
        "weight_index": weights
    }
FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
docker build -t bird-analyzer .
docker run -p 8000:8000 bird-analyzer
GET http://localhost:8000/analyze
# Bird Counting & Weight Estimation

This project detects, tracks, and estimates bird weight proxies from fixed poultry CCTV footage.

## Features
- YOLOv8 + ByteTrack
- Stable bird IDs
- Perspective-aware weight index
- FastAPI inference service
- Annotated output video

## Weight Estimation
Since true weight cannot be measured from monocular video, we compute a
**perspective-corrected visual weight index** using bounding box area,
camera geometry, and temporal smoothing.

## Run Locally
pip install -r requirements.txt
uvicorn app.main:app --reload

## Docker
docker build -t bird-analyzer .
docker run -p 8000:8000 bird-analyzer
