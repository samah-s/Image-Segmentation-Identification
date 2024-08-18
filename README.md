# Object Identification, Segmentation, and Annotation Pipeline

## Overview
This repository provides a comprehensive pipeline for object identification, segmentation, and annotation. The pipeline utilizes various deep learning models for object detection, image captioning, text extraction, and data summarization. The final output includes annotated images with bounding boxes, object descriptions, dominant colors, and a summary table of object attributes.

## Project Structure

.
├── models_dir
│ ├── segmentation_model.py
│ ├── identification_model.py
│ ├── text_extraction_model.py
│ └── summarization_model.py
├── utils_dir
│ ├── paths.py
│ ├── data_mapping.py
│ └── visualization.py
├── input_data
│ ├── input_image.jpg
│ ├── segmented_objects
│ └── master_images
├── output_data
│ ├── final_mapping.json
│ ├── summaries.json
│ └── annotated_images
├── main_pipeline.py
└── README.md



## Pipeline Steps

### 1. Image Segmentation
The first step in the pipeline involves segmenting the input image into individual objects. The segmented objects are saved as separate image files.

```python
segmentation.image_segmentation()

### 2. Object Identification
Each segmented object is then passed through a pre-trained YOLOv5 model to identify the object class.

```python
identification_model.identify_object()

### 3. Text Extraction
Text is extracted from both the segmented objects and the original master images using Tesseract OCR.

```python
text_extraction_model.extract_text_main()
### 4. Object Description Generation
A BLIP model is used to generate descriptions for each object, and the dominant color of each object is also extracted.

```python
summarization_model.generate_summary()
### 5. Data Mapping
The metadata from the object identification, text extraction, and description generation stages are mapped to the corresponding master image.

```python
data_mapping.generate_final_mapping()
### 6. Visualization and Output
Annotated images with bounding boxes, labels, and summary tables are generated and saved.

```python
visualization.final_output()
## Usage
To run the entire pipeline, execute the main_pipeline.py script:

```python
python main_pipeline.py

Ensure that the input images are placed in the input_data directory, and the results will be saved in the output_data directory.

## Dependencies
Python 3.7+
PyTorch
Tesseract OCR
BLIP (Salesforce)
Webcolors
Colorthief
Matplotlib
Pillow
Scikit-learn
Transformers
Others mentioned in requirements.txt
## Installation
To install the required dependencies, you can use the following command:

```bash
pip install -r requirements.txt