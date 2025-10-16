import cv2
import base64
import asyncio
import time
import os
import tempfile
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from deepface import DeepFace

app = FastAPI()

# Allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

camera = None
generic_model = YOLO("yolov8n.pt")  # Default object detection
currency_model = YOLO("best.pt")  # Custom trained model
detection_memory = {}  # {label: (last_seen_time, direction)}
face_recognition_memory = {}  # {name: last_seen_time} - prevents duplicate announcements

# Path to your face database
FACE_DB_PATH = "known_faces"

# Pre-build DeepFace representations on startup (speeds up recognition)
@app.on_event("startup")
async def startup_event():
    if os.path.exists(FACE_DB_PATH):
        print("🔄 Building face database representations...")
        try:
            # This creates embeddings cache for faster recognition
            DeepFace.find(
                img_path=os.path.join(FACE_DB_PATH, os.listdir(FACE_DB_PATH)[0], os.listdir(os.path.join(FACE_DB_PATH, os.listdir(FACE_DB_PATH)[0]))[0]),
                db_path=FACE_DB_PATH,
                model_name="Facenet512",
                enforce_detection=False,
                silent=True
            )
            print("✅ Face database ready")
        except Exception as e:
            print(f"⚠️ Could not pre-build face database: {e}")

def recognize_face(face_img):
    """
    Recognize a face using DeepFace.
    Returns the person's name or 'Unknown'
    """
    try:
        # Save face temporarily
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        cv2.imwrite(temp_file.name, face_img)
        temp_path = temp_file.name
        temp_file.close()
        
        # Find matches in database
        dfs = DeepFace.find(
            img_path=temp_path,
            db_path=FACE_DB_PATH,
            model_name="Facenet512",  # Best performing model
            enforce_detection=False,
            silent=True
        )
        
        # Clean up temp file
        os.unlink(temp_path)
        
        # Process results
        if dfs and len(dfs) > 0 and not dfs[0].empty:
            # Get best match (lowest distance)
            best_match = dfs[0].iloc[0]
            distance = best_match['distance']
            
            # Facenet512 threshold (lower = more similar)
            if distance < 0.4:  # Adjust threshold as needed
                # Extract person name from path
                identity_path = best_match['identity']
                person_name = Path(identity_path).parent.name
                return person_name
        
        return "Unknown"
    
    except Exception as e:
        print(f"⚠️ Face recognition error: {e}")
        return None

@app.websocket("/ws/cam")
async def webcam_feed(websocket: WebSocket):
    global camera, detection_memory, face_recognition_memory
    await websocket.accept()
    try:
        camera = cv2.VideoCapture(0)
        query_params = dict(pair.split('=') for pair in websocket.url.query.split('&')) if websocket.url.query else {}
        model_type = query_params.get("model", "generic")  # Default to generic if not specified

        # Face recognition only works in generic mode
        face_recognition_enabled = model_type == "generic" and os.path.exists(FACE_DB_PATH)
        
        # Frame skip counter for face recognition (process every N frames)
        frame_count = 0
        face_recognition_interval = 10  # Process face recognition every 10 frames

        while True:
            ret, frame = camera.read()
            if not ret:
                break

            frame_count += 1

            # Select model
            model = currency_model if model_type == "currency" else generic_model
            results = model(frame, verbose=False)[0]
            detection = None
            frame_h, frame_w = frame.shape[:2]

            # Draw grid lines for direction
            cv2.line(frame, (frame_w // 3, 0), (frame_w // 3, frame_h), (0, 255, 255), 2)
            cv2.line(frame, (2 * frame_w // 3, 0), (2 * frame_w // 3, frame_h), (0, 255, 255), 2)

            person_detected = False
            person_boxes = []

            for box in results.boxes:
                cls_id = int(box.cls[0])
                label = model.names[cls_id]
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                x_center = (x1 + x2) / 2

                direction = "left" if x_center < frame_w / 3 else "right" if x_center > 2 * frame_w / 3 else "ahead"

                # Check if person is detected
                if label.lower() == "person" and face_recognition_enabled:
                    person_detected = True
                    person_boxes.append((x1, y1, x2, y2, direction))

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"{label} ({direction})",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

                now = time.time()
                if label in detection_memory:
                    last_seen, last_dir = detection_memory[label]
                    if direction == last_dir and now - last_seen >= 2:
                        # Don't announce person detection here, will be replaced by name
                        if label.lower() != "person":
                            detection = f"{label} on {direction}"
                        detection_memory[label] = (now, direction)
                else:
                    detection_memory[label] = (now, direction)

            # Face recognition for detected persons (every N frames to improve performance)
            if person_detected and frame_count % face_recognition_interval == 0:
                now = time.time()
                
                for (x1, y1, x2, y2, direction) in person_boxes:
                    # Expand bbox slightly for better face capture
                    padding = 20
                    x1_crop = max(0, x1 - padding)
                    y1_crop = max(0, y1 - padding)
                    x2_crop = min(frame_w, x2 + padding)
                    y2_crop = min(frame_h, y2 + padding)
                    
                    # Crop face region
                    face_crop = frame[y1_crop:y2_crop, x1_crop:x2_crop]
                    
                    if face_crop.size > 0:
                        # Recognize face
                        person_name = recognize_face(face_crop)
                        
                        if person_name and person_name != "Unknown":
                            # Check if we recently announced this person
                            if person_name not in face_recognition_memory or \
                               now - face_recognition_memory[person_name] >= 10:  # 10 second cooldown
                                detection = f"{person_name} on {direction}"
                                face_recognition_memory[person_name] = now
                                
                                # Draw name on frame
                                cv2.putText(
                                    frame,
                                    f"{person_name}",
                                    (x1, y1 - 30),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.8,
                                    (255, 0, 255),
                                    2,
                                )

            _, buffer = cv2.imencode(".jpg", frame)
            frame_b64 = base64.b64encode(buffer).decode("utf-8")

            payload = {"frame": frame_b64, "detection": detection}
            await websocket.send_json(payload)

            await asyncio.sleep(0.03)  # ~30 fps

    except WebSocketDisconnect:
        print("🔌 Client disconnected cleanly")
    except Exception as e:
        print("⚠️ Error:", e)
    finally:
        if camera and camera.isOpened():
            camera.release()
            camera = None
        cv2.destroyAllWindows()
        if not websocket.client_state.name == "DISCONNECTED":
            await websocket.close()
        print("✅ Camera released, socket closed")
