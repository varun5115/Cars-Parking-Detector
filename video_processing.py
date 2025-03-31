import cv2
import numpy as np
from config import *

def load_video():
    """Load the video and extract the first frame."""
    cap = cv2.VideoCapture(VIDEO_PATH)
    success, first_frame = cap.read()
    if not success:
        raise Exception("Failed to load video.")
    return cap, first_frame


def preprocess_frame(img, val1, val2, val3):
    """Apply grayscale, blur, and thresholding."""
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, GAUSSIAN_BLUR_KERNEL, GAUSSIAN_BLUR_SIGMA)

    # Ensure odd values for adaptive thresholding
    val1 += 1 if val1 % 2 == 0 else 0
    val3 += 1 if val3 % 2 == 0 else 0

    img_thres = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY_INV, val1, val2)
    img_thres = cv2.medianBlur(img_thres, val3)

    kernel = np.ones(DILATION_KERNEL, np.uint8)
    img_thres = cv2.dilate(img_thres, kernel, iterations=DILATION_ITERATIONS)

    return img_thres
