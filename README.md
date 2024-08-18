# Object Identification, Segmentation, and Annotation Pipeline

## Overview
This project involves several steps to perform object detection, segmentation, and analysis on images. The overall workflow includes the following:

* Image Segmentation: Objects within an image are detected and segmented using a pre-trained YOLOv5 model.
* Object Identification: The segmented objects are identified using the same model.
* Text Extraction: Any text present in the images is extracted using Tesseract OCR.
* Object Summarization: The objects are described using a pre-trained BLIP image captioning model. Their dominant colors are also identified.
* Data Mapping: The extracted data (text, object descriptions, and color) is mapped to the original images.
* Visualization: The results are visualized by generating annotated images and summary table



## Pipeline Steps

### 1. Image Segmentation
The first step in the pipeline involves segmenting the input image into individual objects. The segmented objects are saved as separate image files.

```python
segmentation.image_segmentation()
```

### 2. Object Identification
Each segmented object is then passed through a pre-trained YOLOv5 model to identify the object class.

```python
identification_model.identify_object()
```

### 3. Text Extraction
Text is extracted from both the segmented objects and the original master images using Tesseract OCR.

```python
text_extraction_model.extract_text_main()
```
### 4. Object Description Generation
A BLIP model is used to generate descriptions for each object, and the dominant color of each object is also extracted.

```python
summarization_model.generate_summary()
```
### 5. Data Mapping
The metadata from the object identification, text extraction, and description generation stages are mapped to the corresponding master image.

```python
data_mapping.generate_final_mapping()
```
### 6. Visualization and Output
Annotated images with bounding boxes, labels, and summary tables are generated and saved.

```python
visualization.final_output()
```



## Installation
### Prerequisites
Python 3.7+  
PyTorch  
Tesseract OCR (for text extraction)  
Various Python packages (listed in requirements.txt)  

### Setup
1. Clone the repository:
```bash
git clone https://github.com/your-repo-name/project-name.git
```
2. Install the required Python packages:
```bash
pip install -r requirements.txt
```
3. Ensure Tesseract OCR is installed and its path is correctly set in models_dir/text_extraction_model.py.

## Usage

### Running the full pipeline
To execute the entire pipeline, run the following command:
```bash
python main.py
```
This will perform image segmentation, object identification, text extraction, object summarization, data mapping, and visualization

## Outputs
Annotated images and summary tables will be saved in the output directory specified in data/output/ directory.
The final mapping of all data to the master images will be saved in a JSON file, also specified in data/output/ directory.

