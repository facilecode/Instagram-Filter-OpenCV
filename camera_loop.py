import cv2
import numpy as np
from time import time

cam = cv2.VideoCapture(0)

def preprocess(original):
    # You can resize if you want
    #im = cv2.resize(im, (300,200))
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

    # you can play with hyperparameters to adapt to your problem
    gray_smooth = cv2.bilateralFilter(gray, 7, 60, 60)

    outline = cv2.Canny(gray_smooth, 60, 120)

    offset = 10
    padded = np.pad(outline,((0,0),(0,offset)), mode="constant")
    shifted = padded[:,offset:]

    gray_outline = cv2.addWeighted(gray, 1, shifted, 1, 0)

    outline_3_channel = np.stack((shifted,)*3, axis=-1)
    original_outline = cv2.addWeighted(original, 1, outline_3_channel, 1, 0)

    return original_outline

while True:
    """
        bool, ndarray = cam.read()
        if ok=True => OK
        if ok=False => Couldn't capture
    """
    ok, original = cam.read()

    if ok:
        original = preprocess(original)

        cv2.imshow('Outline', original)
        # 1 milisecond pause
        cv2.waitKey(1)
    else:
        print("Error capturing")