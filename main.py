import cv2
import numpy as np

original = cv2.imread('kylie.png')
#im = cv2.resize(im, (300,200))

cv2.imshow('Original', original)
cv2.waitKey(2000) #show during 3 seconds

gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
gray_smooth = cv2.bilateralFilter(gray, 7, 60, 60)
cv2.imshow('Gray Smooth', gray_smooth)
cv2.waitKey(2000) #show during 3 seconds

outline = cv2.Canny(gray_smooth, 60, 120)
cv2.imshow('Outline', outline)
cv2.waitKey(2000) #show during 3 seconds

offset = 10
padded = np.pad(outline,((0,0),(0,offset)), mode="constant")
shifted = padded[:,offset:]
cv2.imshow('Shifted outline', shifted)
cv2.waitKey(2000) #show during 3 seconds

print(f"Gray {gray.shape} - Shifted {shifted.shape}")

gray_outline = cv2.addWeighted(gray, 1, shifted, 1, 0)
cv2.imshow('Gray outline', gray_outline)
cv2.waitKey(2000) #show during 3 seconds

print(f"Original {original.shape} - Shifted {shifted.shape}")

outline_3_channel = np.stack((shifted,)*3, axis=-1)
original_outline = cv2.addWeighted(original, 1, outline_3_channel, 1, 0)
cv2.imshow('Original outline', original_outline)
cv2.waitKey(3000) #show during 3 seconds

