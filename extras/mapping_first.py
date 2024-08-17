import json
import os

# Define the paths
DATA_DIR = 'Documents/personal/completed-projects/ML/ImageSegmentation/data/'
METADATA_FILE = os.path.join(DATA_DIR, 'output/metadata.json')
EXTRACTED_TEXT_FILE = os.path.join(DATA_DIR, 'output/extracted_text_data.json')
SUMMARY_FILE = os.path.join(DATA_DIR, 'output/summary.json')
MAPPED_OUTPUT_FILE = os.path.join(DATA_DIR, 'output/mapped_data.json')

# Function to load JSON data from a file
def load_json(file_path):
    with open(file_path, 'r') as file:
        return [json.loads(line) for line in file] if file_path == METADATA_FILE else json.load(file)

# Function to map data
def map_data(metadata, extracted_text, summaries):
    mapped_data = []
    for entry in metadata:
        unique_id = entry['unique_id']
        image_name = os.path.basename(entry['image_path'])
        summary_entry = next((item for item in summaries if item['unique_id'] == unique_id), {})
        extracted_text_entry = extracted_text.get(image_name, "")
        
        # Combine all relevant information
        mapped_entry = {
            'unique_id': unique_id,
            'master_id': entry['master_id'],
            'image_path': entry['image_path'],
            'bbox': entry['bbox'],
            'confidence': entry['confidence'],
            'class': entry['class'],
            'extracted_text': extracted_text_entry,
            'description': summary_entry.get('description', ""),
            'colors': summary_entry.get('colors', ""),
            'identified_name': summary_entry.get('identified_name', "")
        }
        mapped_data.append(mapped_entry)
    
    return mapped_data

# Function to save the mapped data to a JSON file
def save_mapped_data(mapped_data, output_file):
    with open(output_file, 'w') as file:
        json.dump(mapped_data, file, indent=4)

# Main function
def mapping():
    # Load data from files
    metadata = load_json(METADATA_FILE)
    extracted_text = load_json(EXTRACTED_TEXT_FILE)
    summaries = load_json(SUMMARY_FILE)
    
    # Map the data
    mapped_data = map_data(metadata, extracted_text, summaries)
    
    # Save the mapped data
    save_mapped_data(mapped_data, MAPPED_OUTPUT_FILE)
    print(f"Mapped data saved to {MAPPED_OUTPUT_FILE}")

if __name__ == '__main__':
    mapping()
