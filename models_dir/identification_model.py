import torch
from PIL import Image
import os
import json
from utils_dir.paths import *

# Load the pre-trained YOLOv5 model for object identification
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5l.pt')

def identify_objects(image_dir):
    """
    Identifies and describes objects in the images stored in a given directory.
    
    Args:
        image_dir (str): Directory containing the segmented object images.
        
    Returns:
        dict: A dictionary containing object descriptions keyed by image file names.
    """
    object_descriptions = {}

    for image_file in os.listdir(image_dir):
        if image_file.endswith('.jpg') or image_file.endswith('.png'):
            image_path = os.path.join(image_dir, image_file)

            image = Image.open(image_path)
            
            # Perform object identification
            results = model(image)
            
            labels = results.names  # Class names
            
            # Ensure predictions are available
            if results.pred[0].size(0) > 0:
                classes = results.pred[0][:, -1].cpu().numpy()  # Detected classes

                # Use the most confident prediction
                object_class = labels[int(classes[0])]
                object_descriptions[image_file] = object_class
            else:
                object_descriptions[image_file] = "No object detected"

    return object_descriptions

def save_descriptions(descriptions, output_file):
    """
    Saves the object descriptions to a JSON file.
    
    Args:
        descriptions (dict): Object descriptions.
        output_file (str): Path to the output JSON file.
    """
    with open(output_file, 'w') as f:
        json.dump(descriptions, f, indent=4)

def identify_object():
    # Directory containing segmented object images
    image_dir = SEGMENTED_OBJECTS_DIR

    # Output JSON file to save object descriptions
    output_file = OBJECT_DESCRIPTIONS_FILE
    
    # Identify objects and generate descriptions
    descriptions = identify_objects(image_dir)
    
    # Save descriptions to a file
    save_descriptions(descriptions, output_file)


