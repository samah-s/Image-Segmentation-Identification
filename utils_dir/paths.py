import os

# Define the paths
DATA_DIR = 'Documents/personal/completed-projects/ML/ImageSegmentation/data/'

INPUT_IMAGE_FILE = os.path.join(DATA_DIR, 'input_images/inp_image.jpg')

# Paths to your JSON files
METADATA_FILE = os.path.join(DATA_DIR, 'output/metadata.json')
EXTRACTED_TEXT_FILE = os.path.join(DATA_DIR, 'output/extracted_text_data.json')
SUMMARY_FILE = os.path.join(DATA_DIR, 'output/summary.json')
OBJECT_DESCRIPTIONS_FILE = os.path.join(DATA_DIR, 'output/object_descriptions.json')
FINAL_MAPPING_FILE = os.path.join(DATA_DIR, 'output/final_mapping.json')

#folderr paths
OUTPUT_DIR = os.path.join(DATA_DIR, 'output/')
SEGMENTED_OBJECTS_DIR = os.path.join(DATA_DIR, 'segmented_objects/')
MASTER_IMAGE_DIR = os.path.join(DATA_DIR, 'input_images/')











