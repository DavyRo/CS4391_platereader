from ultralytics import YOLO
import cv2
import pandas as pd
import easyocr

# Initialize OCR reader for license plate text extraction
reader = easyocr.Reader(["en"], gpu=False)

# Save detection results to a CSV file
def save_to_csv(results, output):
    df = pd.DataFrame(results)
    df.to_csv(output, index=False)

# Validate license plate format
def validate_license_plate_format(license_plate_text):
    # Assuming the license plate must have 7 characters
    return len(license_plate_text) == 7

# Read license plate text from a cropped image
def read_license_plate_text(cropped_image):
    detections = reader.readtext(cropped_image)
    for detection in detections:
        _, text, confidence_score = detection
        text = text.upper().replace(" ", "")
        if validate_license_plate_format(text):
            return text, confidence_score
    return None, None

# Load YOLO model for license plate detection
license_plate_detector_model = YOLO("license_plate_detector.pt")

# Load video
video_sample = cv2.VideoCapture("./sample.mp4")

# Process video frames to detect license plates
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
            cropped_image = frame[int(y1):int(y2), int(x1):int(x2)]
            license_plate_text, _ = read_license_plate_text(cropped_image)

            if license_plate_text:  # If a valid license plate text is found
                detection_results.append(
                    {
                        "frame_number": frame_number,
                        "license_number": license_plate_text,
                    }
                )

# Save results to a CSV file
save_to_csv(detection_results, "./results.csv")

# Release video capture
video_sample.release()