import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


def visualize_results(image, boxes, output_path):
    """
    Visualizes the detection results and saves the output image.
    
    Args:
        image (PIL.Image.Image): The original image.
        boxes (tensor): Bounding boxes for detected objects.
        output_path (str): Path to save the visualized image.
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(np.array(image))

    for box in boxes:
        x1, y1, x2, y2, conf, cls = box
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(x1, y1, f'{conf:.2f}', bbox=dict(facecolor='yellow', alpha=0.5), fontsize=12, color='black')

    plt.axis('off')
    plt.savefig(output_path)
    plt.close()
    print(f"Visualized output saved to {output_path}")


