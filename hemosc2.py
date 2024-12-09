import cv2
import pytesseract
import re
import json
import os
import time

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Define the bounding boxes for regions of interest
bounding_boxes = {
    'Blood Pressure': (30, 70, 300, 100),  # (x, y, width, height)
    'Heart Rate': (30, 150, 300, 100),
    'Temperature': (30, 300, 300, 100),
    'Conductivity': (30, 400, 300, 100),
    'Blood Flow Rate': (500, 200, 300, 100),
    'Session Time': (500, 300, 300, 100)
}

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 5)
    contrast_img = cv2.convertScaleAbs(blur, alpha=1.5, beta=0)
    _, thresh = cv2.threshold(contrast_img, 150, 255, cv2.THRESH_BINARY)
    cv2.imwrite('preprocessed_image.png', thresh)
    
    return thresh

def extract_text_from_bounding_boxes(image, bounding_boxes):
    extracted_data = {}

    for field, (x, y, w, h) in bounding_boxes.items():
        roi = image[y:y+h, x:x+w]  # Extract region of interest (ROI)
        
        # Apply OCR on the ROI
        extracted_text = pytesseract.image_to_string(roi)
        
        # Clean and store the extracted text
        extracted_data[field] = extracted_text.strip()
    
    return extracted_data

def process_extracted_data(extracted_data):
    processed_data = {}

    # Extract Blood Pressure (BP)
    if 'Blood Pressure' in extracted_data:
        bp_match = re.search(r'(\d{2,3})/(\d{2,3})', extracted_data['Blood Pressure'])
        if bp_match:
            processed_data['Blood Pressure'] = f"{bp_match.group(1)}/{bp_match.group(2)}"
    
    # Extract Heart Rate (HR)
    if 'Heart Rate' in extracted_data:
        hr_match = re.search(r'(\d+)', extracted_data['Heart Rate'])
        if hr_match:
            processed_data['Heart Rate'] = hr_match.group(1)

    # Extract Temperature
    if 'Temperature' in extracted_data:
        temperature_match = re.search(r'([0-9.]+)', extracted_data['Temperature'])
        if temperature_match:
            processed_data['Temperature'] = temperature_match.group(1)

    # Extract Conductivity
    if 'Conductivity' in extracted_data:
        conductivity_match = re.search(r'([0-9.]+)', extracted_data['Conductivity'])
        if conductivity_match:
            processed_data['Conductivity'] = conductivity_match.group(1)

    # Extract Blood Flow Rate
    if 'Blood Flow Rate' in extracted_data:
        blood_flow_match = re.search(r'([0-9.]+)', extracted_data['Blood Flow Rate'])
        if blood_flow_match:
            processed_data['Blood Flow Rate'] = blood_flow_match.group(1)

    # Extract Session Time
    if 'Session Time' in extracted_data:
        session_time_match = re.search(r'([0-9]{2}:[0-9]{2})', extracted_data['Session Time'])
        if session_time_match:
            processed_data['Session Time'] = session_time_match.group(1)

    return processed_data

def detect_conditions(extracted_data):
    conditions = {}

    if 'Creatinine' in extracted_data and float(extracted_data['Creatinine']) > 1.2:
        conditions['Kidney Failure'] = 'Creatinine increase'
    if 'Creatinine' in extracted_data and float(extracted_data['Creatinine']) > 1.5:
        conditions['Uremia'] = 'Blood urea increase'
    conditions['Metabolic Acidosis'] = 'Lactic acid increase in blood (hypothetical)'
    if 'Conductivity' in extracted_data and (float(extracted_data['Conductivity']) < 13.0 or float(extracted_data['Conductivity']) > 15.0):
        conditions['Electrolyte Imbalance'] = 'Serum potassium, sodium, bicarbonate levels abnormal'

    return conditions

def format_to_json(extracted_data, detected_conditions):
    renalyx_data = {
        "Temperature": extracted_data.get("Temperature", "N/A"),
        "Conductivity": extracted_data.get("Conductivity", "N/A"),
        "Blood Pressure": extracted_data.get("Blood Pressure", "N/A"),
        "Heart Rate": extracted_data.get("Heart Rate", "N/A"),
        "Blood Flow Rate": extracted_data.get("Blood Flow Rate", "N/A"),
        "Session Time": extracted_data.get("Session Time", "N/A"),
        "Creatinine": extracted_data.get("Creatinine", "N/A"),
        "Conditions": detected_conditions
    }
    return json.dumps({"renalyxData": renalyx_data}, indent=4)

if __name__ == "__main__":
    image_path = input("Enter the path to the image of the hemodialysis machine's display: ")
    image_path = r'{}'.format(image_path)
    
    start_time = time.time()

    try:
        # Preprocess the image
        preprocessed_image = preprocess_image(image_path)
        
        # Load preprocessed image for bounding box extraction
        img = cv2.imread('preprocessed_image.png')
        
        # Extract text from bounding boxes
        extracted_data = extract_text_from_bounding_boxes(img, bounding_boxes)
        
        print("\nRaw OCR Extracted Data from Bounding Boxes:")
        print(extracted_data)

        # Process the extracted data
        processed_data = process_extracted_data(extracted_data)

        # Detect any health conditions based on the processed data
        detected_conditions = detect_conditions(processed_data)
        
        # Format data into JSON output
        json_output = format_to_json(processed_data, detected_conditions)

        print("\nExtracted Data in JSON format:")
        print(json_output)

        execution_time = time.time() - start_time
        print(f"\nAlgorithm execution time: {execution_time:.4f} seconds")

    except Exception as e:
        print(f"Error: {e}")
