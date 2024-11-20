# readvideo.py: Opens appropriate video feed and displays it in a window
# Author: Davy Ro

import cv2
import numpy as np

# URL for Zaragoza Bridge
url = 'https://zoocams.elpasozoo.org/BridgeZaragoza3.m3u8'

cap = cv2.VideoCapture(url)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Resize frame to fit screen, match dimensions of real-time feed
    # Original video player resolution is 550x288 on webpage
    # Actual video is thinner (roughly 1/7 of width on each side is white margin)
    # Increase dimensions by 2 to create larger frame
    # Final dimensions: 550*2*5//7 x 288*2
    frameResized = cv2.resize(frame, (550*2*5//7, 288*2))
    cv2.imshow('Video Feed from Zaragoza Bridge (Press q to quit)', frameResized)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()