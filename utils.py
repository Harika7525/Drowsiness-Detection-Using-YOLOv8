from sre_constants import SUCCESS
from ultralytics import YOLO
import streamlit as st
import cv2
import matplotlib.pyplot as plt
import tempfile
from PIL import Image


def display_detected_frames(confidence, model, st_frame, image):
    # Resize the image to a standard size
    image = cv2.resize(image, (720, int(720 * (9 / 16))))

    # predict the object in the image using YOLOv5 model
    result = model.predict(image, conf=confidence)

    # plot the detected objects on the video frame
    res_plotted = result[0].plot()

    st_frame.image(
        res_plotted, caption="Detected Video", channels="BGR", use_column_width=True
    )


@st.cache_resource
def load_model(model_path):
    model = YOLO(model_path)
    return model


def infer_uploaded_image(confidence, model):
    source_image = st.sidebar.file_uploader(
        label="Choose an Image", type=["jpg", "jpeg", "png", "bmp", "webp"]
    )

    col1, col2 = st.columns(2)

    with col1:
        if source_image:
            uploaded_img = Image.open(source_image)

            st.image(
                image=source_image, caption="Uploaded Image", use_column_width=True
            )

    if source_image:
        if st.button("Predict"):
            with st.spinner("Detecting..."):
                result = model.predict(uploaded_img, conf=confidence)
                boxes = result[0].boxes
                result_plotted = result[0].plot()[:, :, ::-1]

                with col2:
                    st.image(
                        result_plotted, caption="Detected Image", use_column_width=True
                    )
                    try:
                        with st.expander("Detection Restults"):
                            for box in boxes:
                                st.write(box.xywh)
                    except Exception as e:
                        st.write("Please upload image first!")
                        st.write(e)


def infer_uploaded_video(confidence, model):
    source_video = st.sidebar.file_uploader(label="Choose a video")

    col1, col2 = st.columns(2)

    if source_video:
        st.video(source_video)

    if source_video:
        if st.button("Predict"):
            with st.spinner("Detecting..."):
                try:
                    tmpfile = tempfile.NamedTemporaryFile()
                    tmpfile.write(source_video.read())
                    vid_cap = cv2.VideoCapture(tmpfile.name)

                    st_frame = st.empty()

                    while vid_cap.isOpened():
                        success, image = vid_cap.read()
                        if success:
                            display_detected_frames(
                                confidence=confidence,
                                model=model,
                                st_frame=st_frame,
                                image=image,
                            )
                        else:
                            vid_cap.release()
                            break
                except Exception as e:
                    st.write(f"Error loading video: {str(e)}")


def infer_uploaded_webcam(confidence, model):
    try:
        flag = st.button(label="Stop Running")
        vid_cap = cv2.VideoCapture(0)
        st_frame = st.empty()
        while not flag:
            success, image = vid_cap.read()
            if success:
                display_detected_frames(
                    confidence=confidence, model=model, st_frame=st_frame, image=image
                )
            else:
                vid_cap.release()
                break
    except Exception as e:
        st.write(f"Error loading webcam video: {str(e)}")
