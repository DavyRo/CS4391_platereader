# readvideo.py: Opens appropriate video feed and displays it in a window
# Author: Davy Ro

from asyncio.windows_events import NULL
import cv2
import numpy as np
import imutils
from ultralytics import YOLO

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
        frame_resized = cv2.resize(frame, (1100, 576))
        # Resize frame in order to remove unnecessary/unreadable video
        frame_resized_cropped = frame_resized[50:400, 100:400]

        # convert to grayscale for easier processing
        # grayScale = cv2.cvtColor(frame_resized_cropped, cv2.COLOR_BGR2GRAY)

        # load YOLO models
        yoloModel = YOLO('yolov8n.pt', verbose=False)
        license_plate_detector = YOLO('license_plate_detector.pt', verbose=False)

        # detect cars
        cars = [2, 3, 5, 7] # class numbers that are vehicles
        detections = yoloModel(frame_resized_cropped, verbose=False)[0]
        carDetections = []
        for d in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, classId = d
            if int(classId) in cars:
                carDetections.append([x1, y1, x2, y2, score])

        # detect license plates
        plateCrop = frame_resized_cropped
        licensePlates = license_plate_detector(frame_resized_cropped, verbose=False)[0]
        for plate in licensePlates.boxes.data.tolist():
            x1, y1, x2, y2, score, classId = plate

            # confirm license plate is connected to car
            xc1, yc1, xc2, yc2, carId = getCar(plate, carDetections)
            if carId != -1:

                # crop around license plate
                temp = frame_resized_cropped[int(y1):int(y2), int(x1):int(x2), :]
                plateCrop.resize(plateCrop, temp.shape)
                plateCrop = temp

        # show image
        cv2.imshow('Video from Stanton Bridge (q to quit)', plateCrop)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # print plate number


    cap.release()
    cv2.destroyAllWindows()
    return

def getCar(plate, cars):
    # initialize values
    x1, y1, x2, y2, score, classId = plate
    found = False

    # get each car in list
    for i in range(len(cars)):
        xc1, yc1, xc2, yc2, carId = cars[i]

        # check if plate is whithin the bounds of the car
        if x1 > xc1 and y1 > yc1 and x2 < xc2 and y2 < yc2:
            carIdx = i
            found = True
            break

    # return car that the license plate is in, if any
    if found:
        return cars[i]
    return -1, -1, -1, -1, -1

if __name__ == '__main__':
    # URL for Zaragoza Bridge
    url = 'https://zoocams.elpasozoo.org/BridgeStanton2.m3u8'
    cap = cv2.VideoCapture(url)
    display_video(cap)
