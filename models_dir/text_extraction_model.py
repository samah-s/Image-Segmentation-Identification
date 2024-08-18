
import pytesseract
from PIL import Image
import os
import json
from utils_dir.paths import *


# Configure the path to the Tesseract executable (adjust as needed)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def extract_text(image_path):
    """
    Extracts text from a given image using Tesseract OCR.
    
    Args:
        image_path (str): Path to the image file.
        
    Returns:
        str: Extracted text from the image.
    """
    try:
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)
        return text.strip()
    except Exception as e:
        print(f"Error extracting text from {image_path}: {e}")
        return ""

def extract_text_from_directory(image_dir, master_image_dir):
    """
    Extracts text from all images in a given directory.
    
    Args:
        image_dir (str): Directory containing the object images.
        
    Returns:
        dict: A dictionary containing extracted text keyed by image file names.
    """
    text_data = {}

    
    for image_file in os.listdir(image_dir):
        if image_file.endswith('.jpg') or image_file.endswith('.png'):
            image_path = os.path.join(image_dir, image_file)
            print(f"Extracting text from {image_file}...")
            text = extract_text(image_path)
            text_data[image_file] = text
    
    for image_file in os.listdir(master_image_dir):
        if image_file.endswith('.jpg') or image_file.endswith('.png'):
            image_path = os.path.join(master_image_dir, image_file)
            print(f"Extracting text from {image_file}...")
            text = extract_text(image_path)
            text_data[image_file] = text
    
    return text_data

def save_text_data(text_data, output_file):
    """
    Saves the extracted text data to a JSON file.
    
    Args:
        text_data (dict): Extracted text data.
        output_file (str): Path to the output JSON file.
    """
    print(f"Saving text data to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(text_data, f, indent=4)
    print("Text data saved.")

def extract_text_main():
    # Directory containing object images
    image_dir = SEGMENTED_OBJECTS_DIR
    master_image_dir = MASTER_IMAGE_DIR
    
    # Output JSON file to save extracted text data
    output_file = EXTRACTED_TEXT_FILE
    
    # Extract text from images and save to file
    text_data = extract_text_from_directory(image_dir, master_image_dir)
    save_text_data(text_data, output_file)
    
    print(f"Text extraction completed. Data saved to {output_file}")


