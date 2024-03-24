# streamlit run main.py
from solutions import blur, img_details, homography, transformations_3x3
import streamlit as st
import cv2


st.set_page_config(
    page_title="OPENCV",
    layout="wide",
)


main_option = st.sidebar.selectbox(
    "Choose main category",
    ("Home", "3x3 transformations", "Blur image", "Homography", "Image debug"),
)


def main_page():
    st.header("Welcome")
    st.write("Choose topic to explore")


image = cv2.cvtColor(cv2.imread("example_images/alien.jpg"), cv2.COLOR_BGR2RGB)
image_room = cv2.cvtColor(cv2.imread("example_images/room.jpg"), cv2.COLOR_BGR2RGB)
segmentation_mask = cv2.imread("example_images/segmentation_mask.png")

if main_option == "Home":
    main_page()

if main_option == "3x3 transformations":
    transformations_3x3.main(image)

if main_option == "Homography":
    homography.main(image, image_room)

if main_option == "Image debug":
    img_details.main(image)

if main_option == "Blur image":
    blur.main(image, segmentation_mask)
