import cv2
import pytesseract
import re
import json
import time

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Define the bounding boxes for regions of interest
bounding_boxes = {
    'Blood Pressure': (250, 380, 100, 50),  # (x, y, width, height) Example values
    'Heart Rate': (450, 300, 100, 50),
    'Temperature': (600, 200, 100, 50),
    'Conductivity': (650, 400, 120, 50)
}

def preprocess_image(image_path):
    """Preprocess the image with grayscale, contrast, and thresholding."""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply sharpening to enhance edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    sharpened = cv2.filter2D(gray, -1, kernel)
    
    # Increase contrast and brightness
    contrast_img = cv2.convertScaleAbs(sharpened, alpha=2.0, beta=50)

    # Adaptive Thresholding
    thresh = cv2.adaptiveThreshold(contrast_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)

    return thresh

def extract_text_from_bounding_boxes(image, bounding_boxes):
    """Extract text from predefined bounding boxes."""
    extracted_data = {}

    for field, (x, y, w, h) in bounding_boxes.items():
        roi = image[y:y+h, x:x+w]  # Extract region of interest (ROI)
        
        # Apply OCR on the ROI with specific config for digits
        extracted_text = pytesseract.image_to_string(roi, config='--psm 7 outputbase digits')
        
        # Clean and store the extracted text
        extracted_data[field] = extracted_text.strip()

        # Display each ROI with bounding boxes for debugging
        cv2.imshow(f'ROI - {field}', roi)
        cv2.waitKey(0)
    
    cv2.destroyAllWindows()
    
    return extracted_data

def process_extracted_data(extracted_data):
    """Process and clean extracted data."""
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

    return processed_data

def detect_conditions(extracted_data):
    """Detect possible medical conditions based on extracted data."""
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
    """Format the extracted data and conditions into JSON."""
    renalyx_data = {
        "Temperature": extracted_data.get("Temperature", "N/A"),
        "Conductivity": extracted_data.get("Conductivity", "N/A"),
        "Blood Pressure": extracted_data.get("Blood Pressure", "N/A"),
        "Heart Rate": extracted_data.get("Heart Rate", "N/A"),
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

        # Save preprocessed image for debugging
        cv2.imwrite("preprocessed_image.png", preprocessed_image)

        # Extract text from bounding boxes
        extracted_data = extract_text_from_bounding_boxes(preprocessed_image, bounding_boxes)

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
