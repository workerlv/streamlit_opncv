import streamlit as st
from . import utils
import numpy as np
import cv2


def blur_area_from_segmentation(rgb_image, segmentation_mask, kernel_size):
    segmentation_mask = cv2.cvtColor(segmentation_mask, cv2.COLOR_BGR2GRAY)
    segmentation_mask_uint8 = (segmentation_mask * 255).astype(np.uint8)
    _, binary_mask = cv2.threshold(segmentation_mask_uint8, 127, 255, cv2.THRESH_BINARY)
    inverted_mask = cv2.bitwise_not(binary_mask)
    blurred_image = cv2.GaussianBlur(rgb_image, (kernel_size, kernel_size), 30, 0)
    result_image = cv2.bitwise_and(rgb_image, rgb_image, mask=inverted_mask)

    result_image_blurred = cv2.bitwise_and(
        blurred_image, blurred_image, mask=binary_mask
    )
    result_image = cv2.add(result_image, result_image_blurred)

    return result_image


def add_blurred_box(image, coordinates, kernel_size):
    x1, y1, x2, y2 = coordinates
    roi = image[y1:y2, x1:x2]
    blurred_roi = cv2.GaussianBlur(roi, (kernel_size, kernel_size), 30)
    image[y1:y2, x1:x2] = blurred_roi
    return image


# def create_segmentation_mask(image_shape):
#     segmentation_mask = np.zeros(image_shape[:2], dtype=np.float32)
#     min_dim = min(image_shape[0], image_shape[1])
#     cv2.circle(
#         segmentation_mask,
#         (int(image_shape[1] / 2), int(image_shape[0] / 2)),
#         int(min_dim * 0.6 / 2),
#         1,
#         thickness=-1,
#     )
#     cv2.imwrite("segmentation_mask.png", segmentation_mask)
#     return segmentation_mask


def sidebar():
    st.sidebar.divider()
    st.sidebar.write("Debug console")

    st.sidebar.divider()
    st.sidebar.write("Resize image so it fits good on your screen")
    image_size_percent = st.sidebar.slider("Image size in percents", 0, 100, 50)

    st.sidebar.divider()
    show_points = st.sidebar.checkbox("Show points in image", True)

    st.sidebar.divider()
    kernel_size = st.sidebar.slider(
        "Kernel size", min_value=3, max_value=99, step=2, value=33
    )

    return image_size_percent, show_points, kernel_size


def main(image, segm_mask):

    image_size_percent, show_points, kernel_size = sidebar()

    st.title("Blur image")

    st.divider()
    st.write("You can upload your own image or use template image")

    uploaded_file = st.file_uploader("Choose your personal image")

    if uploaded_file is not None:
        if utils.is_valid_image(uploaded_file):
            image = utils.process_image(uploaded_file)

    st.divider()

    image_resized = utils.resize_image(image, image_size_percent)

    # simple blur
    st.subheader("Blur square from image")

    cols_1, cols_2 = st.columns(2)

    img_room_width = image_resized.shape[1]
    img_room_height = image_resized.shape[0]

    with cols_1:
        x_coordinates = st.slider("x coordinates", 0, img_room_width, (100, 400))

    with cols_2:
        y_coordinates = st.slider("y coordinates", 0, img_room_height, (160, 280))

    point_d1 = (x_coordinates[0], y_coordinates[0])
    point_d2 = (x_coordinates[1], y_coordinates[1])

    if show_points:
        image_resized = utils.draw_dots_on_image(
            image_resized,
            [point_d1, point_d2],
        )

    add_blurred_box(
        image_resized,
        [x_coordinates[0], y_coordinates[0], x_coordinates[1], y_coordinates[1]],
        kernel_size,
    )

    st.image(image_resized)

    # blur from mask
    st.divider()
    st.subheader("Blur image from mask")
    st.write("Image mask size must be same as image size")
    st.write("Mask values must be from 0 till 1")

    uploaded_mask_file = st.file_uploader("Choose your mask")

    if uploaded_mask_file is not None:
        if utils.is_valid_image(uploaded_mask_file):
            segm_mask = utils.process_image(uploaded_mask_file)

    if segm_mask.max() != 1:
        st.warning("Segmentation mask values must be from 0 till 1")
        return

    if image.shape[0] != segm_mask.shape[0] or image.shape[1] != segm_mask.shape[1]:
        st.warning("Image and segmentation dimensions must be equal....")
        return

    col_a, col_b = st.columns(2)

    with col_a:
        st.write("Mask")
        st.image(segm_mask * 255)

    with col_b:
        st.write("Final result")
        image_w_segm_mask = blur_area_from_segmentation(image, segm_mask, kernel_size)
        st.image(image_w_segm_mask)
