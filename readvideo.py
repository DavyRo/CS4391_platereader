# readvideo.py: Opens appropriate video feed and displays it in a window
# Author: Davy Ro

import cv2
import numpy as np

def display_video(cap):
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize frame to fit screen, match dimensions of real-time feed
        # Original video player resolution is 550x288 on webpage
        # Actual video is thinner (roughly 1/7 of width on each side is white margin)
        # Increase dimensions by 2 to create larger frame
        # Final dimensions: 550*2*5//7 x 288*2
        frame_resized = cv2.resize(frame, (785, 576))
        # Resize frame in order to remove unnecessary/unreadable video
        frame_resized_cropped = frame_resized[200:576, 200:600]
        cv2.imshow('Video from Zaragoza Bridge (q to quit)', frame_resized_cropped)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return


if __name__ == '__main__':
    # URL for Zaragoza Bridge
    url = 'https://zoocams.elpasozoo.org/BridgeZaragoza3.m3u8'
    cap = cv2.VideoCapture(url)
    display_video(cap)