# streamlit run scripts/main.py
import transformations.transformations_3x3.transformations_3x3 as transform3x3
import transformations.homography.homography as homography
import streamlit as st
import cv2


st.set_page_config(
    page_title="OPENCV",
    layout="wide",
)


main_option = st.selectbox(
    "Choose main category",
    (
        "Main page",
        "3x3 transformations",
        "Homography",
    ),
)


def main_page():
    st.header("Welcome")
    st.write("Choose topic to explore")


image = cv2.cvtColor(cv2.imread("example_images/alien.jpg"), cv2.COLOR_BGR2RGB)
image_room = cv2.cvtColor(cv2.imread("example_images/room.jpg"), cv2.COLOR_BGR2RGB)

if main_option == "Main page":
    main_page()

if main_option == "3x3 transformations":
    transform3x3.main(image)

if main_option == "Homography":
    homography.main(image, image_room)
