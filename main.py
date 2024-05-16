import streamlit as st
import numpy as np
from ultralytics import YOLO
import cv2 as cv
import os 
import tempfile

model = YOLO("yolov8x")

st.title("Object Detection Using YOLOv8x")

# Upload image
uploaded_image = st.file_uploader("Upload your image", type=['png', 'jpg', 'jpeg'])

if uploaded_image:
    st.image(uploaded_image, caption="Original Image", use_column_width=True)
    
    # Converting the image into a numpy array and then into the OpenCV format of image
    nparr = np.frombuffer(uploaded_image.getbuffer(), np.uint8)
    img_opencv = cv.imdecode(nparr, cv.IMREAD_COLOR)

    # Predict using YOLOv8x model
    results = model.predict(img_opencv, save=False)
    
    # Draw boxes on the image
    for detection in results[0].boxes:
        x, y, w, h = detection.xywh[0]  # Extracting x, y, w, h from detection.xywh
        x, y, w, h = map(float, (x, y, w, h))  # Convert values to float
        x1, y1 = int(x - w / 2), int(y - h / 2)  # Calculate top-left corner
        x2, y2 = int(x + w / 2), int(y + h / 2)  # Calculate bottom-right corner
        cv.rectangle(img_opencv, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Convert BGR to RGB
    img_opencv_rgb = cv.cvtColor(img_opencv, cv.COLOR_BGR2RGB)

    st.image(img_opencv_rgb, caption="Processed Image", use_column_width=True)

    # Save the processed image to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
        temp_file_path = temp_file.name
        cv.imwrite(temp_file_path, img_opencv_rgb)

    st.download_button(label="Download", data=open(temp_file_path, "rb"), file_name="processed_image.png")
