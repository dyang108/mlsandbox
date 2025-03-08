import speech_recognition as sr
import requests
import json
import cv2
import torch
import queue
import threading
import time
from PIL import Image
from transformers import DetrForObjectDetection, DetrImageProcessor
from collections import defaultdict
mic_name = "Brio 101: USB Audio (hw:3,0)"
mic_list = sr.Microphone.list_microphone_names()
device_index = mic_list.index(mic_name)

tracked_objects = defaultdict(lambda: {"box": None, "score": 0, "last_seen": 0})
PERSISTENCE_TIME = 1.5  # Keep objects visible for 1.5 seconds after last detection

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "deepseek-r1:1.5b"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# COCO labels mapping for DETR
COCO_LABELS = {1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 5: "airplane",
    6: "bus", 7: "train", 8: "truck", 9: "boat", 10: "traffic light", 11: "fire hydrant",
    13: "stop sign", 14: "parking meter", 15: "bench", 16: "bird", 17: "cat", 18: "dog",
    19: "horse", 20: "sheep", 21: "cow", 22: "elephant", 23: "bear", 24: "zebra", 25: "giraffe",
    27: "backpack", 28: "umbrella", 31: "handbag", 32: "tie", 33: "suitcase", 34: "frisbee",
    35: "skis", 36: "snowboard", 37: "sports ball", 38: "kite", 39: "baseball bat",
    40: "baseball glove", 41: "skateboard", 42: "surfboard", 43: "tennis racket",
    44: "bottle", 46: "wine glass", 47: "cup", 48: "fork", 49: "knife", 50: "spoon",
    51: "bowl", 52: "banana", 53: "apple", 54: "sandwich", 55: "orange", 56: "broccoli",
    57: "carrot", 58: "hot dog", 59: "pizza", 60: "donut", 61: "cake", 62: "chair",
    63: "couch", 64: "potted plant", 65: "bed", 67: "dining table", 70: "toilet", 72: "tv",
    73: "laptop", 74: "mouse", 75: "remote", 76: "keyboard", 77: "cell phone", 78: "microwave",
    79: "oven", 80: "toaster", 81: "sink", 82: "refrigerator", 84: "book", 85: "clock",
    86: "vase", 87: "scissors", 88: "teddy bear", 89: "hair drier", 90: "toothbrush"
}

def load_object_detection_model():
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50").to(device)
    return processor, model

def detect_objects(frame, processor, model, target_label=None):
    image = Image.fromarray(cv2.resize(frame, (320, 240)))  # Reduce resolution
    inputs = processor(images=image, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    target_scores = outputs.logits.softmax(-1)[0, :, :-1]
    target_boxes = outputs.pred_boxes[0]
    detected_objects = []
    
    for score, box in zip(target_scores, target_boxes):
        max_score, label = score.max(0)
        label_id = label.item()
        label_name = COCO_LABELS.get(label_id, "unknown")
        
        if max_score.item() > 0.5:
            x_center, y_center, width, height = box.cpu().detach().numpy()
            x1 = int((x_center - width / 2) * frame.shape[1])
            y1 = int((y_center - height / 2) * frame.shape[0])
            x2 = int((x_center + width / 2) * frame.shape[1])
            y2 = int((y_center + height / 2) * frame.shape[0])
            detected_objects.append((label_name, (x1, y1, x2, y2), max_score.item()))
    
    return detected_objects

def recognize_speech(speech_queue):
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    with microphone as source:
        recognizer.adjust_for_ambient_noise(source)

    while True:
        with sr.Microphone(device_index=device_index) as source:
            print("Listening for a command...")
            try:
                audio = recognizer.listen(source, timeout=5)
                text = recognizer.recognize_google(audio).lower()
                speech_queue.put(text)
                print(f"Recognized: {text}")
            except sr.UnknownValueError:
                print("Could not understand audio.")
            except sr.RequestError:
                print("Speech recognition request failed.")
            except Exception as e:
                print(f"Speech recognition error: {e}")

def open_photo_booth():
    processor, model = load_object_detection_model()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Photo booth active. Say an object name to highlight it. Say 'exit' to stop.")
    target_label = None
    last_capture_time = time.time()
    
    speech_queue = queue.Queue()
    speech_thread = threading.Thread(target=recognize_speech, args=(speech_queue,))
    speech_thread.daemon = True
    speech_thread.start()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        if time.time() - last_capture_time >= .1:  # Process frame every 100ms
            detected_objects = detect_objects(frame, processor, model, target_label)
            last_capture_time = time.time()
            update_tracked_objects(detected_objects)

            # for label, box, score in detected_objects:
            #     x1, y1, x2, y2 = box
            #     color = (0, 0, 255) if target_label and target_label == label else (0, 255, 0)
            #     cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            #     cv2.putText(frame, f"{label}: {score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        draw_tracked_objects(frame, target_label)
        cv2.imshow("Photo Booth - Object Detection", frame)

        if not speech_queue.empty():
            spoken_text = speech_queue.get()
            if spoken_text == "exit":
                break
            else:
                target_label = spoken_text
                print(f"Target object set to: {target_label}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def draw_tracked_objects(frame, target_label=None):
    """ Draw persistent bounding boxes """
    for label, data in tracked_objects.items():
        x1, y1, x2, y2 = data["box"]
        score = data["score"]

        # Highlight target object in red, others in green
        color = (0, 0, 255) if target_label and target_label == label else (0, 255, 0)
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{label}: {score:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def update_tracked_objects(detected_objects):
    current_time = time.time()

    for label, box, score in detected_objects:
        tracked_objects[label]["box"] = box
        tracked_objects[label]["score"] = score
        tracked_objects[label]["last_seen"] = current_time  # Update last seen time

    # Remove stale objects
    for label in list(tracked_objects.keys()):
        if current_time - tracked_objects[label]["last_seen"] > PERSISTENCE_TIME:
            del tracked_objects[label]


if __name__ == "__main__":
    open_photo_booth()
