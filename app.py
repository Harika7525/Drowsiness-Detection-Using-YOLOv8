import os
from pathlib import Path

from ultralytics import YOLO
from utils import (
    load_model,
    infer_uploaded_image,
    infer_uploaded_video,
    infer_uploaded_webcam,
)

from Drowsiness_Detection.constant.application import APP_HOST, APP_PORT


import streamlit as st


# setting page layout
st.set_page_config(
    page_title="Drowsiness Detection using YOLOv5",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Home Page Heading
st.title("Drowsiness Detection using YOLOv5")

# sidebar
st.sidebar.header("DL Model Config")


# model options
task_type = st.sidebar.selectbox("Select Task", ["Detection"])


confidence = float(st.sidebar.slider("Select Model Confidence", 30, 100, 50)) / 100

model_path = os.path.join("best.pt")


# load pretrained DL model
try:
    model = YOLO(model_path)
except Exception as e:
    st.error(f"Unable to load model. Please check the specified path: {model_path}")
    st.error(f"Exception: {e}")

# Source
SOURCES_LIST = ["Image", "Video", "Webcam"]

# image/video options
st.sidebar.header("Image/Video Config")
source_selectbox = st.sidebar.selectbox("Select Source", SOURCES_LIST)

source_img = None

if source_selectbox == SOURCES_LIST[0]:  # Image
    infer_uploaded_image(confidence, model)
elif source_selectbox == SOURCES_LIST[1]:  # Video
    infer_uploaded_video(confidence, model)
elif source_selectbox == SOURCES_LIST[2]:  # Webcam
    infer_uploaded_webcam(confidence, model)
else:
    st.error("Currently only 'Image' and 'Video' source are implemented")
