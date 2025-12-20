import cv2
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from yolov5 import YOLOv5
from sort import Sort  # You'll need to install SORT: pip install filterpy (SORT is built on it)

# Load YOLOv5 model (pre-trained on COCO, which includes birds)
model = YOLOv5('yolov5s.pt')  # Small model for speed; use 'yolov5l.pt' for accuracy

# Initialize SORT tracker
tracker = Sort()

# Video input (replace with your CCTV video path)
video_path = 'sample_poultry_video.mp4'  # Example: Download a short poultry video
cap = cv2.VideoCapture(video_path)

# Output data structures
frame_data = []  # List of dicts: {'frame': int, 'timestamp': float, 'count': int, 'weights': list}
bird_weights = {}  # Track weights per bird ID

# Weight estimation proxy: Scale bounding box area to a weight index
def estimate_weight(bbox):
    # bbox: [x1, y1, x2, y2]
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    area = width * height
    # Arbitrary scaling: Assume average bird area ~10,000 pixels corresponds to 2kg
    weight_proxy = (area / 10000) * 2  # In kg units
    return max(weight_proxy, 0.1)  # Min weight to avoid negatives

frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000  # In seconds
    
    # Detect birds (class 14 in COCO is 'bird')
    results = model.predict(frame, classes=[14])  # Filter for birds only
    detections = []
    for result in results.xyxy[0]:  # [x1, y1, x2, y2, conf, class]
        x1, y1, x2, y2, conf, cls = result
        if conf > 0.5:  # Confidence threshold
            detections.append([x1, y1, x2, y2, conf])
    
    detections = np.array(detections)
    
    # Track detections
    if len(detections) > 0:
        tracked_objects = tracker.update(detections)
    else:
        tracked_objects = tracker.update(np.empty((0, 5)))
    
    # Process tracked birds
    current_weights = []
    for track in tracked_objects:
        x1, y1, x2, y2, track_id = track
        weight = estimate_weight([x1, y1, x2, y2])
        bird_weights[track_id] = weight  # Update weight for this ID
        current_weights.append(weight)
    
    # Record data
    frame_data.append({
        'frame': frame_count,
        'timestamp': timestamp,
        'count': len(tracked_objects),
        'weights': current_weights
    })
    
    # Optional: Visualize (comment out for headless)
    for track in tracked_objects:
        x1, y1, x2, y2, track_id = track
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f'ID: {int(track_id)}', (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Process and output results
df = pd.DataFrame(frame_data)
df.to_csv('bird_counts_and_weights.csv', index=False)

# Plot counts over time
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(df['timestamp'], df['count'])
plt.title('Bird Counts Over Time')
plt.xlabel('Time (s)')
plt.ylabel('Count')

# Plot average weight over time
df['avg_weight'] = df['weights'].apply(lambda x: np.mean(x) if x else 0)
plt.subplot(1, 2, 2)
plt.plot(df['timestamp'], df['avg_weight'])
plt.title('Average Bird Weight Proxy Over Time')
plt.xlabel('Time (s)')
plt.ylabel('Weight Proxy (kg)')
plt.tight_layout()
plt.savefig('bird_analysis_plots.png')
plt.show()

# Summary
total_birds = df['count'].max()
avg_weight = df['avg_weight'].mean()
print(f"Total unique birds tracked: {len(bird_weights)}")
print(f"Average weight proxy: {avg_weight:.2f} kg")
print("Outputs saved: bird_counts_and_weights.csv and bird_analysis_plots.png")