import cv2
import pickle
import cvzone
import numpy as np
from config import *


# Initialize background subtractor (MOG2 is more robust to shadows)
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)
def apply_background_subtraction(frame):
    """Apply background subtraction to the frame."""
    fg_mask = bg_subtractor.apply(frame)
    
    # Clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    
    return fg_mask

def load_parking_spaces():
    """Load parking spaces from file or create an empty list."""
    try:
        with open(PARKING_FILE, 'rb') as f:
            loaded_list = pickle.load(f)
            pos_list = [(x, y, w, h) if len(pos) == 4 else (pos[0], pos[1], 50, 50)
                        for pos in loaded_list if len(pos) in {2, 4}]
            print(f"Loaded {len(pos_list)} valid parking spaces.")
    except:
        pos_list = []
    return pos_list


def save_parking_spaces(pos_list):
    """Save parking spaces to file."""
    with open(PARKING_FILE, 'wb') as f:
        pickle.dump(pos_list, f)


def draw_parking_spaces(img, pos_list, free_spaces):
    """Draw the parking spaces on the image."""
    for pos in pos_list:
        x, y, w, h = pos
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)

    # Display free/total spaces
    cvzone.putTextRect(img, f'Free: {free_spaces}/{len(pos_list)}', (50, 60), thickness=3, offset=20, colorR=(0, 200, 0))


def check_spaces(img, bg_mask, pos_list):
    """Check for free/occupied spaces using background subtraction mask."""
    free_spaces = 0

    for pos in pos_list:
        x, y, w, h = pos
        img_crop = bg_mask[y:y + h, x:x + w]
        count = cv2.countNonZero(img_crop)

        # More precise parking threshold
        if count < PARKING_THRESHOLD:
            color = (0, 200, 0)  # Green = Free
            free_spaces += 1
        else:
            color = (0, 0, 200)  # Red = Occupied

        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, str(count), (x + 5, y + h - 6), cv2.FONT_HERSHEY_PLAIN, 1, color, 2)

    return free_spaces


def check_spaces_with_contours(img, img_thres, pos_list):
    """Check for free/occupied spaces using contour detection."""
    free_spaces = 0

    for pos in pos_list:
        x, y, w, h = pos
        img_crop = img_thres[y:y + h, x:x + w]

        # Find contours
        contours, _ = cv2.findContours(img_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Count valid contours
        occupied = False
        for contour in contours:
            area = cv2.contourArea(contour)

            # Detect large areas (likely cars)
            if area > 1500:  # Adjust the area threshold based on car size
                occupied = True
                break

        if occupied:
            color = (0, 0, 200)  # Red = Occupied
        else:
            color = (0, 200, 0)  # Green = Free
            free_spaces += 1

        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

    return free_spaces


try:
    from ultralytics import YOLO
    # Load YOLOv8 model - will be loaded only if we use the ML method
    model = None
except ImportError:
    print("Warning: ultralytics not installed. ML-based detection will not work.")
    model = None

def load_yolo_model():
    """Load the YOLO model if not already loaded"""
    global model
    if model is None:
        try:
            print("Loading YOLO model...")
            model = YOLO("yolov8n.pt")
            print("YOLO model loaded successfully")
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            return False
    return True

def check_spaces_with_yolo(img, pos_list):
    """Use YOLOv8 for vehicle detection and parking space occupancy."""
    free_spaces = 0
    
    # Ensure YOLO model is loaded
    if not load_yolo_model():
        return 0
    
    # Run detection on the entire frame
    results = model(img)[0]
    
    # Get all detected vehicles (cars, trucks, buses, motorcycles)
    vehicle_classes = [2, 3, 5, 7]  # COCO classes for car, motorcycle, bus, truck
    vehicle_boxes = []
    
    for box in results.boxes.data:
        x1, y1, x2, y2, conf, cls = box
        if int(cls) in vehicle_classes and conf > 0.5:
            vehicle_boxes.append((int(x1), int(y1), int(x2), int(y2)))
    
    # Check each parking space for overlapping with detected vehicles
    for pos in pos_list:
        x, y, w, h = pos
        parking_area = (x, y, x + w, y + h)
        
        # Check if any vehicle overlaps with this parking space
        is_occupied = False
        for v_x1, v_y1, v_x2, v_y2 in vehicle_boxes:
            # Calculate IoU (Intersection over Union)
            intersection_area = max(0, min(v_x2, parking_area[2]) - max(v_x1, parking_area[0])) * \
                               max(0, min(v_y2, parking_area[3]) - max(v_y1, parking_area[1]))
            
            parking_area_size = w * h
            overlap_ratio = intersection_area / parking_area_size
            
            # If overlap is significant, mark as occupied
            if overlap_ratio > 0.15:  # Threshold for considering a space occupied
                is_occupied = True
                break
        
        # Draw rectangle based on occupancy
        if is_occupied:
            color = (0, 0, 200)  # Red = Occupied
        else:
            color = (0, 200, 0)  # Green = Free
            free_spaces += 1
            
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
    
    return free_spaces
