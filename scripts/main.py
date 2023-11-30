# streamlit run scripts/main.py
import transformations.transformations_3x3.transformations_3x3 as transform3x3
import streamlit as st
import cv2

st.set_page_config(
    page_title="OPENCV",
    layout="wide", 
)

# TODO: whene multiple categories added then add this
# main_option = st.selectbox(
#     "Choose main category",
#     ("Main page", "Transformations"))

# TODO: whene more categories added then add this
# sub_category = st.selectbox(
#     "Choose sub-category",
#     ("3x3 tranformations")
# )

# if main_option == "Transformations":
image = cv2.imread("alien.jpg")
transform3x3.main(image)