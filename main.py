import cv2
from config import *
from video_processing import load_video, preprocess_frame
from parking_logic import load_parking_spaces, save_parking_spaces, draw_parking_spaces, check_spaces

# --- Initialize Video and Parking Spaces ---
cap, first_frame = load_video()
pos_list = load_parking_spaces()

# --- Variables for mouse interaction ---
start_point, end_point = None, None
drawing = False

# --- GUI Parameter Tuning ---
cv2.namedWindow(WINDOW_NAME)
cv2.resizeWindow(WINDOW_NAME, *WINDOW_SIZE)
cv2.createTrackbar("Val1", WINDOW_NAME, 25, 50, lambda x: None)
cv2.createTrackbar("Val2", WINDOW_NAME, 16, 50, lambda x: None)
cv2.createTrackbar("Val3", WINDOW_NAME, 5, 50, lambda x: None)


def mouse_callback(event, x, y, flags, params):
    """Mouse click-and-drag event handler for marking spaces."""
    global start_point, end_point, drawing, pos_list

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_point = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        end_point = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        end_point = (x, y)

        # Add new parking space
        if start_point and end_point:
            x1, y1 = start_point
            x2, y2 = end_point
            pos_list.append((min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1)))
            save_parking_spaces(pos_list)


# --- Main Loop ---
mode = "marking" if len(pos_list) == 0 else "detection"

while True:
    if mode == "marking":
        img = first_frame.copy()
        cv2.setMouseCallback("Image", mouse_callback)

        # Draw current parking spaces
        draw_parking_spaces(img, pos_list, 0)

        # Draw ongoing selection
        if drawing and start_point and end_point:
            cv2.rectangle(img, start_point, end_point, (0, 255, 0), 2)

        cv2.imshow("Image", img)

        key = cv2.waitKey(1)
        if key == 27 or key == ord('q'):
            mode = "detection"

    elif mode == "detection":
        success, img = cap.read()
        if not success:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # Pre-processing
        val1 = cv2.getTrackbarPos("Val1", WINDOW_NAME)
        val2 = cv2.getTrackbarPos("Val2", WINDOW_NAME)
        val3 = cv2.getTrackbarPos("Val3", WINDOW_NAME)

        img_thres = preprocess_frame(img, val1, val2, val3)
        free_spaces = check_spaces(img, img_thres, pos_list)
        draw_parking_spaces(img, pos_list, free_spaces)

        cv2.imshow("Image", img)

        key = cv2.waitKey(1)
        if key == 27 or key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
