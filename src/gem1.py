import cv2
from ultralytics import YOLO
import supervision as sv

# 1. Model Load
model = YOLO('best.pt') 

# 2. Video Setup
cap = cv2.VideoCapture("video.mp4")
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = int(cap.get(cv2.CAP_PROP_FPS))

out = cv2.VideoWriter('moving_camera_count5_trim.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

# 3. Visuals Setup (Bareek lines aur chote dots)
box_annotator = sv.BoxAnnotator(color=sv.Color.from_hex('#00FF00'), thickness=1)

# Counting Set: Isme sirf unique IDs save hongi
counted_ids = set()

print("Moving Camera Tracking started... Press 'q' to stop.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Tracking Logic
    results = model.track(
        frame, 
        persist=True, 
        conf=0.35,      # Thora high confidence taake ghalat count na ho
        iou=0.5, 
        tracker="bytetrack.yaml",
        imgsz=640
    )[0]
    
    detections = sv.Detections.from_ultralytics(results)
    
    if results.boxes.id is not None:
        ids = results.boxes.id.cpu().numpy().astype(int)
        detections.tracker_id = ids
        
        # Unique ID Logic (Set duplicates filter kar deta hai)
        for obj_id in ids:
            counted_ids.add(obj_id)

        # Draw green boxes
        frame = box_annotator.annotate(scene=frame, detections=detections)

        # Draw white dots
        for xyxy in detections.xyxy:
            x_c = int((xyxy[0] + xyxy[2]) / 2)
            y_c = int((xyxy[1] + xyxy[3]) / 2)
            cv2.circle(frame, (x_c, y_c), 3, (255, 255, 255), -1)

    # 4. White UI Panel
    cv2.rectangle(frame, (0, 0), (250, 90), (255, 255, 255), -1)
    
    current_count = len(detections) 
    total_unique = len(counted_ids) # Yeh variable theek kar diya gaya hai
    
    cv2.putText(frame, f"Detected: {current_count}", (15, 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 200), 2)
    cv2.putText(frame, f"Total Count: {total_unique}", (15, 75), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 0, 0), 2)

    out.write(frame)
    cv2.imshow("Moving Camera Counter", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Final Unique Count: {len(counted_ids)}")