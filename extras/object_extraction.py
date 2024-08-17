import os
import numpy as np
import torch
from PIL import Image
import cx_Oracle  # Ensure cx_Oracle is installed for Oracle DB interaction

# Database connection details
dsn = cx_Oracle.makedsn('samah', '1521', service_name = 'ORCL')
connection = cx_Oracle.connect(user='system', password='password', dsn=dsn)
cursor = connection.cursor()

# Ensure output directories exist
output_dir = 'Documents/personal/completed-projects/ML/ImageSegmentation/objects/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def extract_and_save_objects(image_np, boxes, master_id):
    metadata = []
    for i, box in enumerate(boxes):
        x1, y1, x2, y2, conf, cls = box
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Extract object from image
        object_image = image_np[y1:y2, x1:x2]
        object_image_pil = Image.fromarray(object_image)
        
        # Generate unique ID for the object
        unique_id = f'{master_id}_{i}'
        
        # Save object image
        object_image_path = os.path.join(output_dir, f'{unique_id}.jpg')
        object_image_pil.save(object_image_path)
        
        # Save object image and metadata to the database
        save_object_to_db(unique_id, master_id, object_image_path, x1, y1, x2, y2, conf, cls)
        
        # Append metadata
        metadata.append({
            'unique_id': unique_id,
            'master_id': master_id,
            'bbox': [x1, y1, x2, y2],
            'confidence': float(conf),
            'class': int(cls)
        })
    
    return metadata

def save_object_to_db(unique_id, master_id, image_path, x1, y1, x2, y2, confidence, cls):
    with open(image_path, 'rb') as file:
        image_data = file.read()
    
    # Insert metadata and image into the database
    sql = """INSERT INTO image_metadata (unique_id, master_id, image, x1, y1, x2, y2, confidence, class) 
             VALUES (:unique_id, :master_id, :image, :x1, :y1, :x2, :y2, :confidence, :class)"""
    cursor.execute(sql, {
        'unique_id': unique_id,
        'master_id': master_id,
        'image': image_data,
        'x1': x1,
        'y1': y1,
        'x2': x2,
        'y2': y2,
        'confidence': confidence,
        'class': cls
    })
    connection.commit()

def main():
    # Load the pre-trained YOLOv5 model for object detection
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

    # Load your image
    image_path = 'Documents/personal/completed-projects/ML/ImageSegmentation/dog_man_image.jpg'
    image = Image.open(image_path)
    image_np = np.array(image)

    # Process the input image
    results = model(image)

    # Get bounding boxes
    boxes = results.xyxy[0].cpu().numpy()  # Bounding boxes in the format [x1, y1, x2, y2, conf, cls]

    # Define master ID for the original image
    master_id = os.path.basename(image_path).split('.')[0]

    # Extract objects and save to database
    metadata = extract_and_save_objects(image_np, boxes, master_id)

    # Print or save metadata as needed
    import json
    metadata_path = 'Documents/personal/completed-projects/ML/ImageSegmentation/metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)

    print(f'Saved {len(metadata)} objects and metadata to the database and {metadata_path}')

if __name__ == "__main__":
    main()

# Close the database connection
cursor.close()
connection.close()
