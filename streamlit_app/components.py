import streamlit as st
import pandas as pd
import json
from PIL import Image
import os
from utils_dir.paths import *


def upload_image():
    uploaded_file = st.file_uploader("Upload an image", type=["jpg"])
    return uploaded_file

def display_segmented_objects_image():
    annotated_image_path = os.path.join(OUTPUT_DIR, '.png')
    if os.path.exists(annotated_image_path):
        annotated_image = Image.open(annotated_image_path)
        st.image(annotated_image, caption='Segmented Objects Image', use_column_width=True)
    else:
        st.write("SEgmented objects image not found.")
    
    

def display_segmented_objects(image_path):
    segmented_objects = []
    for file_name in os.listdir(SEGMENTED_OBJECTS_DIR):
        if file_name.endswith(('.jpg', '.png')):
            file_path = os.path.join(SEGMENTED_OBJECTS_DIR, file_name)
            segmented_objects.append({
                'unique_id': file_name.split('.')[0],
                'image': Image.open(file_path)
            })
    return segmented_objects

def display_segmented_objects_details(image_path):
    segmented_objects = []
    with open(os.path.join(OUTPUT_DIR, 'metadata.json'), 'r') as file:
        metadata = [json.loads(line) for line in file]
    with open(os.path.join(OUTPUT_DIR, 'object_descriptions.json'), 'r') as file:
        descriptions = json.load(file)
    for file_name in os.listdir(SEGMENTED_OBJECTS_DIR):
        if file_name.endswith(('.jpg', '.png')):
            file_path = os.path.join(SEGMENTED_OBJECTS_DIR, file_name)
            segmented_objects.append({
                'unique_id': file_name.split('.')[0],
                'image': Image.open(file_path)
            })
    details = []
    for obj in metadata:
        unique_id = obj['unique_id']
        description = descriptions.get(f"{unique_id}.jpg", "")
        details.append({
            'Unique ID': unique_id,
            'Description': description,
            'Bounding Box': obj['bbox'],
            'Confidence': obj['confidence'],
            'Class': obj['class']
        })
    return segmented_objects



def show_object_details():
    # Load metadata and object descriptions
    with open(os.path.join(OUTPUT_DIR, 'metadata.json'), 'r') as file:
        metadata = [json.loads(line) for line in file]
    
    with open(os.path.join(OUTPUT_DIR, 'object_descriptions.json'), 'r') as file:
        descriptions = json.load(file)
    

    summary_path = os.path.join(OUTPUT_DIR, 'final_mapping.json')
    if os.path.exists(summary_path):
        with open(summary_path, 'r') as file:
            summary = json.load(file)
        
        # Flatten the JSON structure
        flattened_data = []
        master_image = summary["inp_image"]
        master_image_path = master_image["image_path"]
        extracted_text = master_image["extracted_text"]
        
        for obj in master_image["objects"]:
            obj_data = {
                "Unique ID": obj["unique_id"],
                "Description": obj.get("description", ""),
                "Identified Name": obj.get("identified_name", ""),
                "Colors": obj.get("colors", ""),
                "Bounding Box": f"({obj['bbox'][0]}, {obj['bbox'][1]}, {obj['bbox'][2]}, {obj['bbox'][3]})",
                "Confidence": obj["confidence"],
                "Class": obj["class"]
                
            }
            flattened_data.append(obj_data)

    return flattened_data



def display_final_output():
    annotated_image_path = os.path.join(OUTPUT_DIR, 'inp_image_annotated.jpg')
    if os.path.exists(annotated_image_path):
        annotated_image = Image.open(annotated_image_path)
        st.image(annotated_image, caption='Final Annotated Image', use_column_width=True)
    else:
        st.write("Annotated image not found.")
    
    summary_path = os.path.join(OUTPUT_DIR, 'final_mapping.json')
    if os.path.exists(summary_path):
        with open(summary_path, 'r') as file:
            summary = json.load(file)
        
        # Flatten the JSON structure
        flattened_data = []
        master_image = summary["inp_image"]
        master_image_path = master_image["image_path"]
        extracted_text = master_image["extracted_text"]
        
        for obj in master_image["objects"]:
            obj_data = {
                "Unique ID": obj["unique_id"],
                "Description": obj.get("description", ""),
                "Identified Name": obj.get("identified_name", ""),
                "Colors": obj.get("colors", ""),
                "Bounding Box": f"({obj['bbox'][0]}, {obj['bbox'][1]}, {obj['bbox'][2]}, {obj['bbox'][3]})",
                "Confidence": obj["confidence"],
                "Class": obj["class"],                
                "Object Image Path": obj["image_path"],
                
            }
            flattened_data.append(obj_data)
        
        # Convert the flattened data into a DataFrame
        summary_df = pd.DataFrame(flattened_data)
        return summary_df, master_image_path, extracted_text
    else:
        st.write("Final mapping not found.")
        return pd.DataFrame(), master_image_path, extracted_text

def display_summary_table(summary):
    if not summary.empty:
        st.write("Final Mapping Table:")
        st.dataframe(summary)
    else:
        st.write("No data available for summary.")
