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
        frame_resized = cv2.resize(frame, (1100, 576))
        #cv2.imshow('Video from Stanton Bridge (q to quit)', frame_resized)
        # Resize frame in order to remove unnecessary/unreadable video
        frame_resized_cropped = frame_resized[100:400, 100:400]
        cv2.imshow('Video from Stanton Bridge (q to quit)', frame_resized_cropped)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return


if __name__ == '__main__':
    # URL for Stanton Bridge
    url = 'https://zoocams.elpasozoo.org/BridgeStanton2.m3u8'
    cap = cv2.VideoCapture(url)
    display_video(cap)