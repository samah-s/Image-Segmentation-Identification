import streamlit as st
import os
import pandas as pd
from PIL import Image
import streamlit_app.components as components
import models_dir.segmentation_model as segmentation
import models_dir.identification_model as identification_model
import models_dir.text_extraction_model as text_extraction_model
import models_dir.summarization_model as summarization_model
import utils_dir.data_mapping as data_mapping
import utils_dir.visualization as visualization
from utils_dir.paths import *

# Ensure directories exist
os.makedirs(MASTER_IMAGE_DIR, exist_ok=True)
os.makedirs(SEGMENTED_OBJECTS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. File Upload
uploaded_file = components.upload_image()

def del_dir_files(directory):

        # List all files in the directory
        files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

        # Delete each file
        for file in files:
            file_path = os.path.join(directory, file)
            os.remove(file_path)
            print(f"Deleted: {file_path}")

    

if uploaded_file is not None:
    del_dir_files(MASTER_IMAGE_DIR)
    del_dir_files(OUTPUT_DIR)
    del_dir_files(SEGMENTED_OBJECTS_DIR)

    # Save the uploaded image in the MASTER_IMAGE_DIR
    image = Image.open(uploaded_file)
    image_path = INPUT_IMAGE_FILE
    image.save(image_path)

    # Perform the pipeline processing steps
    segmentation.image_segmentation()
    identification_model.identify_object()
    text_extraction_model.extract_text_main()
    summarization_model.generate_summary()
    data_mapping.generate_final_mapping()
    visualization.final_output()

    # 2. Segmentation Display
    st.write("Segmented Objects:")
    components.display_segmented_objects_image()

    st.markdown("---")
    segmented_objects = components.display_segmented_objects(image_path)
    object_details = components.show_object_details()
    for i in range(len(segmented_objects)):
        obj = segmented_objects[i]
        detail = pd.DataFrame([object_details[i]])
        st.image(obj['image'], caption=f"Segmented Object {obj['unique_id']}")
        st.dataframe(detail)

    # 3. Object Details
    # st.write("Object Details:")

    # 4. Final Output
    st.markdown("---")
    st.write("Final Output:")
    summary, master_path, extracted_text = components.display_final_output()

    st.dataframe({"Master Image": "inp_image.jpg","Extracted Text":extracted_text,"Master Image Path":master_path})

    # 5. Summary Table
    components.display_summary_table(summary)


    

    

        

