import json
import os

# Paths to your JSON files
METADATA_FILE = 'Documents/personal/completed-projects/ML/ImageSegmentation/data/output/metadata.json'
EXTRACTED_TEXT_FILE = 'Documents/personal/completed-projects/ML/ImageSegmentation/data/output/extracted_text_data.json'
SUMMARY_FILE = 'Documents/personal/completed-projects/ML/ImageSegmentation/data/output/summary.json'
MASTER_IMAGE_PATH = "Documents/personal/completed-projects/ML/ImageSegmentation/data/input_images/"

def load_json_data(file_path):
    print(f"Loading data from {file_path}...")
    with open(file_path, 'r') as file:
        data = [json.loads(line) for line in file] if file_path == METADATA_FILE else json.load(file)
    print(f"Data loaded from {file_path}.")
    return data



def map_data_to_master_image(metadata, extracted_text, summary):
    # Dictionary to hold all master image mappings
    master_image_mapping = {}
    
    print("Mapping data to master images...")
    
    # Iterate over metadata entries to create the mapping
    for entry in metadata:
        master_id = entry['master_id']
        print(f"Processing master_id: {master_id}")
        
        # Initialize the master image mapping if not already done
        if master_id not in master_image_mapping:
            master_image_mapping[master_id] = {
                'image_path': MASTER_IMAGE_PATH+master_id+".jpg",  # master image path
                'extracted_text': extracted_text.get(f"{master_id}.jpg", ""),  # extracted text
                'objects': []
            }
            print(f"Initialized mapping for master_id: {master_id}")
        
        # Add object features to the corresponding master image
        object_summary = next((obj for obj in summary if obj['unique_id'] == entry['unique_id']), {})
        if object_summary:
            print(f"Adding object with unique_id: {entry['unique_id']} to master_id: {master_id}")
            master_image_mapping[master_id]['objects'].append({
                'unique_id': entry['unique_id'],
                'image_path': entry['image_path'],
                'bbox': entry['bbox'],
                'confidence': entry['confidence'],
                'class': entry['class'],
                'description': object_summary.get('description', ''),
                'identified_name': object_summary.get('identified_name', ''),
                'colors': object_summary.get('colors', 'Unknown')
            })
    
    print("Data mapping completed.")
    return master_image_mapping

def generate_final_mapping():
    metadata = load_json_data(METADATA_FILE)
    extracted_text = load_json_data(EXTRACTED_TEXT_FILE)
    summary = load_json_data(SUMMARY_FILE)
    print("Generating final mapping...")
    final_mapping = map_data_to_master_image(metadata, extracted_text, summary)
    
    # Output the final mapping
    output_file = 'Documents/personal/completed-projects/ML/ImageSegmentation/data/output/final_mapping.json'
    with open(output_file, 'w') as file:
        json.dump(final_mapping, file, indent=4)
    
    print(f"Final mapping written to {output_file}.")


