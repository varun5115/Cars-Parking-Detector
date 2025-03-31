import cv2
import pickle
import cvzone
import numpy as np

PARKING_THRESHOLD = 500


# --- File and Video Setup ---
video_path = 'car_parking.mp4'
parking_file = 'CarParkPos'

# Load video and extract the first frame for marking
cap = cv2.VideoCapture(video_path)
success, first_frame = cap.read()
if not success:
    print("Failed to load video.")
    exit()

# --- Load Existing Parking Data ---
try:
    with open(parking_file, 'rb') as f:
        loaded_list = pickle.load(f)
        
        # Ensure all positions use (x, y, w, h) format
        posList = [(x, y, w, h) if len(pos) == 4 else (pos[0], pos[1], 50, 50) 
                   for pos in loaded_list if len(pos) in {2, 4}]
        print(f"Loaded {len(posList)} valid parking spaces.")
except:
    posList = []

# --- Variables for Mouse Interaction ---
start_point, end_point = None, None
drawing = False

# --- Window for parameter tuning ---
cv2.namedWindow("Vals")
cv2.resizeWindow("Vals", 640, 240)
cv2.createTrackbar("Val1", "Vals", 25, 50, lambda x: None)
cv2.createTrackbar("Val2", "Vals", 16, 50, lambda x: None)
cv2.createTrackbar("Val3", "Vals", 5, 50, lambda x: None)

# --- Mouse Callback Function for Click-and-Drag Selection ---
def mouseClick(event, x, y, flags, params):
    global start_point, end_point, drawing, posList

    if event == cv2.EVENT_LBUTTONDOWN:  # Start dragging
        drawing = True
        start_point = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE and drawing:  # Dragging
        end_point = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:  # Stop dragging
        drawing = False
        end_point = (x, y)

        # Save the parking space with click-and-drag dimensions
        if start_point and end_point:
            x1, y1 = start_point
            x2, y2 = end_point
            posList.append((min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1)))

            # Save updated parking spots
            with open(parking_file, 'wb') as f:
                pickle.dump(posList, f)

# --- Function to Check Free and Occupied Spaces ---
def checkSpaces(img):
    spaces = 0

    # Pre-processing
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 1)

    val1 = cv2.getTrackbarPos("Val1", "Vals")
    val2 = cv2.getTrackbarPos("Val2", "Vals")
    val3 = cv2.getTrackbarPos("Val3", "Vals")
    
    # Ensure odd values for adaptive thresholding
    val1 += 1 if val1 % 2 == 0 else 0
    val3 += 1 if val3 % 2 == 0 else 0

    imgThres = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, val1, val2)
    imgThres = cv2.medianBlur(imgThres, val3)
    
    kernel = np.ones((3, 3), np.uint8)
    imgThres = cv2.dilate(imgThres, kernel, iterations=1)

    # Check each parking space
    for pos in posList:
        try:
            x, y, w, h = pos
            imgCrop = imgThres[y:y + h, x:x + w]
            count = cv2.countNonZero(imgCrop)

            # Threshold for free/occupied space
            if count < PARKING_THRESHOLD:
                color = (0, 200, 0)  # Green = Free
                thic = 5
                spaces += 1
            else:
                color = (0, 0, 200)  # Red = Occupied
                thic = 2

            cv2.rectangle(img, (x, y), (x + w, y + h), color, thic)
            cv2.putText(img, str(count), (x + 5, y + h - 6), cv2.FONT_HERSHEY_PLAIN, 1, color, 2)

        except Exception as e:
            print(f"Error processing position {pos}: {e}")
            continue

    # Display free/total parking spaces
    cvzone.putTextRect(img, f'Free: {spaces}/{len(posList)}', (50, 60), thickness=3, offset=20, colorR=(0, 200, 0))

# --- Main Loop ---
mode = "marking" if len(posList) == 0 else "detection"

while True:
    if mode == "marking":
        img = first_frame.copy()
        cv2.setMouseCallback("Image", mouseClick)

        # Draw current polygons
        for pos in posList:
            x, y, w, h = pos
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)

        # Draw live selection while dragging
        if drawing and start_point and end_point:
            cv2.rectangle(img, start_point, end_point, (0, 255, 0), 2)

        cv2.imshow("Image", img)

        key = cv2.waitKey(1)
        if key == 27 or key == ord('q'):  # Press 'ESC' or 'q' to finish marking
            mode = "detection"

    elif mode == "detection":
        success, img = cap.read()
        if not success:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        checkSpaces(img)
        cv2.imshow("Image", img)

        key = cv2.waitKey(1)
        if key == 27 or key == ord('q'):  # Press 'ESC' or 'q' to exit
            break

cap.release()
cv2.destroyAllWindows()
