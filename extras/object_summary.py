import os
import torch
from PIL import Image
import numpy as np
import json
import clip
from torchvision import transforms
from colorthief import ColorThief

# Define paths
DATA_DIR = 'Documents/personal/completed-projects/ML/ImageSegmentation/data/'
OUTPUT_DIR = os.path.join(DATA_DIR, 'segmented_objects/')
SUMMARY_FILE = os.path.join(DATA_DIR, 'output/object_summaries.json')

# Ensure the output directory exists
os.makedirs(os.path.dirname(SUMMARY_FILE), exist_ok=True)

# Function to load the CLIP model
def load_clip_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess, device

# Function to summarize object color
def get_dominant_color(image_path):
    color_thief = ColorThief(image_path)
    dominant_color = color_thief.get_color(quality=1)
    return dominant_color

# Function to determine if the object is living or non-living
def determine_living_status(description):
    living_keywords = ['person', 'animal', 'plant', 'insect', 'bird']
    if any(keyword in description for keyword in living_keywords):
        return "Living"
    else:
        return "Non-Living"

# Function to determine if the object is typically found indoors or outdoors
def determine_environment(description):
    outdoor_keywords = ['tree', 'car', 'street', 'park', 'grass', 'sky']
    indoor_keywords = ['table', 'chair', 'bed', 'sofa', 'lamp', 'television']
    
    if any(keyword in description for keyword in outdoor_keywords):
        return "Outdoor"
    elif any(keyword in description for keyword in indoor_keywords):
        return "Indoor"
    else:
        return "Unknown"

# Function to generate object summary
def generate_summary(model, preprocess, device, image_path):
    image = Image.open(image_path)
    image_input = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        text_inputs = torch.cat([clip.tokenize(f"This is a {label}") for label in ['person', 'animal', 'object', 'plant', 'furniture', 'vehicle']]).to(device)
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_inputs)

        logits_per_image, logits_per_text = model(image_input, text_inputs)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        best_label_idx = np.argmax(probs)

    description = f"{['person', 'animal', 'object', 'plant', 'furniture', 'vehicle'][best_label_idx]}"
    dominant_color = get_dominant_color(image_path)
    living_status = determine_living_status(description)
    environment = determine_environment(description)

    summary = {
        "description": description,
        "dominant_color": dominant_color,
        "living_status": living_status,
        "environment": environment
    }
    return summary

# Function to summarize all objects in a directory
def summarize_objects_in_directory(output_dir):
    model, preprocess, device = load_clip_model()
    summaries = {}

    for image_file in os.listdir(output_dir):
        if image_file.endswith('.jpg') or image_file.endswith('.png'):
            image_path = os.path.join(output_dir, image_file)
            summary = generate_summary(model, preprocess, device, image_path)
            summaries[image_file] = summary

    with open(SUMMARY_FILE, 'w') as f:
        json.dump(summaries, f, indent=4)

    print(f"Summarized attributes saved to {SUMMARY_FILE}")


