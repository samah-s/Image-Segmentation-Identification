import os
import json
from PIL import Image
import numpy as np
import torch
from sklearn.cluster import KMeans
from transformers import BlipProcessor, BlipForConditionalGeneration
import webcolors

# Define the paths
DATA_DIR = 'Documents/personal/completed-projects/ML/ImageSegmentation/data/'
OUTPUT_DIR = os.path.join(DATA_DIR, 'segmented_objects/')
SUMMARY_FILE = os.path.join(DATA_DIR, 'output/summary.json')
METADATA_FILE = os.path.join(DATA_DIR, 'output/metadata.json')
OBJECT_DESCRIPTIONS_FILE = os.path.join(DATA_DIR, 'output/object_descriptions.json')

# Load the image captioning model
def load_captioning_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

# Generate description for a single image
def generate_description(image_path, processor, model):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(image, return_tensors="pt")
    out = model.generate(**inputs)
    description = processor.decode(out[0], skip_special_tokens=True)
    return description

from colorthief import ColorThief
import webcolors

def closest_color(requested_color):
    min_colors = {}
    for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_color[0]) ** 2
        gd = (g_c - requested_color[1]) ** 2
        bd = (b_c - requested_color[2]) ** 2
        min_colors[(rd + gd + bd)] = name
    return min_colors[min(min_colors.keys())]

def get_color_name(rgb_tuple):
    try:
        # Convert RGB to hex
        hex_value = webcolors.rgb_to_hex(rgb_tuple)
        # Get the color name directly
        return webcolors.hex_to_name(hex_value)
    except ValueError:
        # If exact match not found, find the closest color
        return closest_color(rgb_tuple)

def get_dominant_color(image_path):
    color_thief = ColorThief(image_path)
    dominant_color = color_thief.get_color(quality=1)
    return dominant_color

# Extract dominant color from an image and convert to color name
def extract_dominant_color_name(image_path):
    
    rgb_color = get_dominant_color(image_path)
    color_name = get_color_name((rgb_color[0], rgb_color[1], rgb_color[2]))
    return color_name


# Load object descriptions from file
def load_object_descriptions(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

# Summarize object attributes
def summarize_object_attributes(metadata_file, descriptions_file):
    processor, model = load_captioning_model()
    summaries = []
    descriptions = load_object_descriptions(descriptions_file)

    with open(metadata_file, 'r') as f:
        metadata = [json.loads(line) for line in f]

    for entry in metadata:
        image_path = entry['image_path']
        description = generate_description(image_path, processor, model)
        colors = extract_dominant_color_name(image_path)
        object_id = descriptions.get(os.path.basename(image_path), "Unknown Object")

        if entry['confidence'] < 0.5:
            description = ""  # Modify if needed

        summaries.append({
            'unique_id': entry['unique_id'],
            'description': description,
            'bbox': entry['bbox'],
            'confidence': entry['confidence'],
            'class': entry['class'],
            'colors': colors,
            'identified_name': object_id
        })

    # Save summaries to a JSON file
    with open(SUMMARY_FILE, 'w') as f:
        json.dump(summaries, f, indent=4)

# Main function
def generate_summary():
    summarize_object_attributes(METADATA_FILE, OBJECT_DESCRIPTIONS_FILE)
    print(f'Summaries have been saved to {SUMMARY_FILE}')


