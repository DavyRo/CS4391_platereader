from ultralytics import YOLO
import cv2
import numpy as np
import pandas as pd
import string
import easyocr
from scipy.interpolate import interp1d

# Initialize the OCR reader
reader = easyocr.Reader(['en'], gpu=False)

# Mapping dictionaries for character conversion
dict_char_to_int = {'O': '0',
                    'I': '1',
                    'J': '3',
                    'A': '4',
                    'G': '6',
                    'S': '5'}

dict_int_to_char = {'0': 'O',
                    '1': 'I',
                    '3': 'J',
                    '4': 'A',
                    '6': 'G',
                    '5': 'S'}

def write_csv(results, output_path):
    """
    Write the results to a CSV file.
    """
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)

def license_complies_format(text):
    """
    Check if the license plate text complies with the required format.
    """
    if len(text) != 7:
        return False
    return all([
        (text[i] in string.ascii_uppercase or text[i] in dict_int_to_char.keys()) if i in [0, 1, 4, 5, 6] else
        (text[i] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[i] in dict_char_to_int.keys())
        for i in range(7)
    ])

def format_license(text):
    """
    Format the license plate text by converting characters using mapping dictionaries.
    """
    license_plate_ = ''
    mapping = {0: dict_int_to_char, 1: dict_int_to_char, 4: dict_int_to_char, 5: dict_int_to_char, 6: dict_int_to_char,
               2: dict_char_to_int, 3: dict_char_to_int}
    for j in range(7):
        license_plate_ += mapping[j].get(text[j], text[j])
    return license_plate_

def read_license_plate(license_plate_crop):
    """
    Read the license plate text from a cropped image.
    """
    detections = reader.readtext(license_plate_crop)
    for detection in detections:
        bbox, text, score = detection
        text = text.upper().replace(' ', '')
        if license_complies_format(text):
            return format_license(text), score
    return None, None

def interpolate_bounding_boxes(data):
    """
    Interpolates missing bounding boxes for vehicles and license plates across frames.
    """
    frame_numbers = np.array([int(row['frame_nmr']) for row in data])
    car_ids = np.array([int(float(row['car_id'])) for row in data])
    car_bboxes = np.array([list(map(float, row['car_bbox'][1:-1].split())) for row in data])
    license_plate_bboxes = np.array([list(map(float, row['license_plate_bbox'][1:-1].split())) for row in data])

    interpolated_data = []
    unique_car_ids = np.unique(car_ids)
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

                if frame_number - prev_frame_number > 1:
                    frames_gap = frame_number - prev_frame_number
                    x = np.array([prev_frame_number, frame_number])
                    x_new = np.linspace(prev_frame_number, frame_number, num=frames_gap, endpoint=False)
                    car_bboxes_interpolated.extend(interp1d(x, np.vstack((prev_car_bbox, car_bbox)), axis=0)(x_new)[1:])
                    license_plate_bboxes_interpolated.extend(interp1d(x, np.vstack((prev_license_plate_bbox, license_plate_bbox)), axis=0)(x_new)[1:])

            car_bboxes_interpolated.append(car_bbox)
            license_plate_bboxes_interpolated.append(license_plate_bbox)

        for i, frame_number in enumerate(range(car_frame_numbers[0], car_frame_numbers[-1] + 1)):
            interpolated_data.append({
                'frame_nmr': frame_number,
                'car_id': car_id,
                'car_bbox': car_bboxes_interpolated[i],
                'license_plate_bbox': license_plate_bboxes_interpolated[i],
            })

    return interpolated_data

# Load models
coco_model = YOLO('yolov8n.pt')
license_plate_detector = YOLO('license_plate_detector.pt')

# Load video
cap = cv2.VideoCapture('./sample.mp4')

results = []
frame_nmr = -1
ret = True

while ret:
    frame_nmr += 1
    ret, frame = cap.read()
    if ret:
        license_plates = license_plate_detector(frame)[0]
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, _, _ = license_plate
            license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2)]
            license_plate_text, _ = read_license_plate(license_plate_crop)
            results.append({'frame_nmr': frame_nmr, 'license_number': license_plate_text or ''})

write_csv(results, './test.csv')
print("Results saved to test.csv")