from ultralytics import YOLO
import cv2
import numpy as np
import pandas as pd
import easyocr
from scipy.interpolate import interp1d

# Initialize OCR reader for license plate text extraction
reader = easyocr.Reader(["en"], gpu=True)

# Mapping dictionaries for character conversion
char_to_num = {"O": "0", "I": "1", "J": "3", "A": "4", "G": "6", "S": "5"}
num_to_char = {"0": "O", "1": "I", "3": "J", "4": "A", "6": "G", "5": "S"}


# Save detection results to a CSV file
def save_to_csv(results, output):

    df = pd.DataFrame(results)
    df.to_csv(output, index=False)


# Validate license plate format
def validate_license_plate_format(license_plate_text):

    if (
        len(license_plate_text) != 7
    ):  # Assuming the license plate must have 7 characters
        return False

    return all(
        [
            # Check if characters at certain positions are letters or mapped values
            (
                (license_plate_text[i] in "ABCDEFGHIJKLMNOPQRSTUVWXYZ")
                or (license_plate_text[i] in num_to_char.keys())
                if i in [0, 1, 4, 5, 6]
                # Check if characters at other positions are digits or mapped values
                else (
                    (license_plate_text[i] in "0123456789")
                    or (license_plate_text[i] in char_to_num.keys())
                )
            )
            for i in range(7)
        ]
    )


# Format license plate text based on character mappings
def format_license_plate_text(license_plate_text):

    formatted_license_plate = ""

    # Define mapping rules for each character position
    mapping = {
        0: num_to_char,
        1: num_to_char,
        2: char_to_num,
        3: char_to_num,
        4: num_to_char,
        5: num_to_char,
        6: num_to_char,
    }

    # Apply mapping to each character position
    for i in range(7):
        formatted_license_plate += mapping[i].get(
            license_plate_text[i], license_plate_text[i]
        )

    return formatted_license_plate


# Read license plate text from a cropped image
def read_license_plate_text(cropped_image):

    detections = reader.readtext(cropped_image)

    for detection in detections:
        _, text, confidence_score = detection
        text = text.upper().replace(" ", "")

        if validate_license_plate_format(text):
            return format_license_plate_text(text), confidence_score

    return None, None


# Interpolate missing bounding boxes for smooth tracking
def interpolate_missing_bounding_boxes(data):

    frame_numbers = np.array([int(row["frame_number"]) for row in data])

    car_ids = np.array([int(float(row["car_id"])) for row in data])

    car_bboxes = np.array(
        [list(map(float, row["car_bbox"][1:-1].split())) for row in data]
    )

    license_plate_bboxes = np.array(
        [list(map(float, row["license_plate_bbox"][1:-1].split())) for row in data]
    )

    interpolated_data = []
    unique_car_ids = np.unique(car_ids)

    # Process each unique car ID
    for car_id in unique_car_ids:
        car_mask = car_ids == car_id
        car_frame_numbers = frame_numbers[car_mask]
        car_bboxes_interpolated = []
        license_plate_bboxes_interpolated = []

        for i in range(len(car_bboxes[car_mask])):
            frame_number = car_frame_numbers[i]
            car_bbox = car_bboxes[car_mask][i]
            license_plate_bbox = license_plate_bboxes[car_mask][i]

            if i > 0:
                prev_frame_number = car_frame_numbers[i - 1]
                prev_car_bbox = car_bboxes_interpolated[-1]
                prev_license_plate_bbox = license_plate_bboxes_interpolated[-1]

                # Handle frame gaps
                if frame_number - prev_frame_number > 1:
                    frames_gap = frame_number - prev_frame_number
                    x = np.array([prev_frame_number, frame_number])
                    x_new = np.linspace(
                        prev_frame_number, frame_number, num=frames_gap, endpoint=False
                    )

                    # Interpolate car and license plate bounding boxes
                    car_bboxes_interpolated.extend(
                        interp1d(x, np.vstack((prev_car_bbox, car_bbox)), axis=0)(
                            x_new
                        )[1:]
                    )
                    license_plate_bboxes_interpolated.extend(
                        interp1d(
                            x,
                            np.vstack((prev_license_plate_bbox, license_plate_bbox)),
                            axis=0,
                        )(x_new)[1:]
                    )

            # Append bounding boxes
            car_bboxes_interpolated.append(car_bbox)
            license_plate_bboxes_interpolated.append(license_plate_bbox)

        # Add interpolated data
        for i, frame_number in enumerate(
            range(car_frame_numbers[0], car_frame_numbers[-1] + 1)
        ):
            interpolated_data.append(
                {
                    "frame_number": frame_number,
                    "car_id": car_id,
                    "car_bbox": car_bboxes_interpolated[i],
                    "license_plate_bbox": license_plate_bboxes_interpolated[i],
                }
            )

    return interpolated_data


# Load YOLO models for vehicle and license plate detection
vehicle_detector_model = YOLO("yolov8n.pt")
license_plate_detector_model = YOLO("license_plate_detector.pt")

# Load video
video_sample = cv2.VideoCapture("./sample.mp4")

# Process video frames to detect vehicles and license plates
detection_results = []
frame_number = -1
has_frame = True

while has_frame:
    frame_number += 1
    has_frame, frame = video_sample.read()

    if has_frame:
        # Detect license plates in the frame
        license_plates = license_plate_detector_model(frame)[0]

        # Process each detected license plate
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, _, _ = license_plate
            cropped_image = frame[int(y1) : int(y2), int(x1) : int(x2)]
            license_plate_text, _ = read_license_plate_text(cropped_image)
            detection_results.append(
                {
                    "frame_number": frame_number,
                    "license_number": license_plate_text or "",
                }
            )

# Save results to a CSV file
save_to_csv(detection_results, "./results.csv")