import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# Path to the final mapping JSON
FINAL_MAPPING_FILE = 'Documents/personal/completed-projects/ML/ImageSegmentation/data/output/final_mapping.json'

def load_final_mapping(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def draw_bbox(ax, bbox, label, color='red'):
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    rect = patches.Rectangle((x1, y1), width, height, linewidth=2, edgecolor=color, facecolor='none')
    ax.add_patch(rect)
    ax.text(x1, y1 - 5, label, color=color, fontsize=10, weight='bold')

def plot_image_with_annotations(master_image_data, output_dir):
    image_path = master_image_data['image_path']
    image = Image.open(image_path)
    
    # Set up the plot
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    # Draw bounding boxes and annotations for each object
    for obj in master_image_data['objects']:
        bbox = obj['bbox']
        label = f"{obj['identified_name']} ({obj['unique_id']})"
        draw_bbox(ax, bbox, label, color='blue')

    # Save the annotated image
    output_image_path = os.path.join(output_dir, f'{os.path.basename(image_path).split(".")[0]}_annotated.jpg')
    plt.axis('off')
    plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)  # Close the figure after saving

import matplotlib.pyplot as plt
import os

def generate_summary_table(master_image_data, output_dir):
    # Create a table of object data
    table_data = []
    for obj in master_image_data['objects']:
        table_data.append([
            obj['unique_id'],
            obj['identified_name'],
            obj['description'],
            f"{obj['confidence']:.2f}",
            obj['class'],
            obj['colors']
        ])
    
    # Plot table
    fig, ax = plt.subplots(figsize=(12, len(table_data) * 0.6))  # Increased figure size for clarity
    ax.axis('tight')
    ax.axis('off')
    
    # Create and customize table
    table = ax.table(
        cellText=table_data,
        colLabels=['Unique ID', 'Identified Name', 'Description', 'Confidence', 'Class', 'Colors'],
        cellLoc='center',
        loc='center',
        cellColours=[[None]*6 for _ in table_data],  # Optional: Customize cell colors if needed
        colColours=['lightgrey']*6  # Optional: Customize column header background color
    )
    
    # Set font size
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    
    # Auto adjust column widths
    table.auto_set_column_width([i for i in range(len(table_data[0]))])
    
    # Save the table as an image with higher resolution
    image_path = master_image_data['image_path']
    output_table_path = os.path.join(output_dir, f'{os.path.basename(image_path).split(".")[0]}_summary_table.jpg')
    plt.savefig(output_table_path, bbox_inches='tight', pad_inches=0.1, dpi=300)  # Increased DPI to 300
    plt.close(fig)  # Close the figure after saving


def final_output():
    final_mapping = load_final_mapping(FINAL_MAPPING_FILE)

    # Output directory for annotated images and tables
    output_dir = 'Documents/personal/completed-projects/ML/ImageSegmentation/data/output/'

    for master_id, master_image_data in final_mapping.items():
        print(f"Processing master image: {master_id}")

        # Generate the annotated image
        plot_image_with_annotations(master_image_data, output_dir)
        
        # Generate the summary table
        generate_summary_table(master_image_data, output_dir)

if __name__ == '__main__':
    final_output()
