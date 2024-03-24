import streamlit as st
from . import utils
import numpy as np
import cv2


def sidebar():
    st.sidebar.divider()
    st.sidebar.write("Debug console")

    st.sidebar.divider()
    st.sidebar.write("Resize image so it fits good on your screen")

    image_size_percent = st.sidebar.slider(
        "Foreground image size in percents", 0, 100, 10
    )
    image_size_percent2 = st.sidebar.slider(
        "Background image size in percents", 0, 100, 10
    )

    return image_size_percent, image_size_percent2


def overlay_nonrectangular_region(background, overlay, points):
    points_array = np.array(points, dtype=np.int32)
    points_array = points_array.reshape((4, 2))
    points_array = [points_array[1], points_array[0], points_array[2], points_array[3]]
    points_array = np.array(points_array, dtype=np.int32)
    mask = np.zeros_like(overlay)
    cv2.fillPoly(mask, [points_array], (255, 255, 255))
    region_to_overlay = cv2.bitwise_and(overlay, mask)
    inverse_mask = cv2.bitwise_not(mask)
    background_without_overlay = cv2.bitwise_and(background, inverse_mask)
    result_image = cv2.add(background_without_overlay, region_to_overlay)

    return result_image


def homography_transformation(image, image2, src_points, dst_points):
    img_1 = image.copy()
    img_2 = image2.copy()

    homography_matrix, _ = cv2.findHomography(src_points, dst_points)

    transformed_h_image = cv2.warpPerspective(
        img_1, homography_matrix, (img_2.shape[1], img_2.shape[0])
    )

    combined_image = overlay_nonrectangular_region(
        img_2, transformed_h_image, dst_points
    )

    return combined_image


def main(image, room_image):

    image_size_percent, image_size_percent2 = sidebar()

    st.title("Homography")

    st.divider()
    st.write("You can upload your own image or use template image")

    col1, col2 = st.columns(2)

    with col1:
        uploaded_file = st.file_uploader("Choose your foreground image")

        if uploaded_file is not None:
            if utils.is_valid_image(uploaded_file):
                image = utils.process_image(uploaded_file)

        st.divider()
        st.write("Foreground image")
        image_resized = utils.resize_image(image, image_size_percent)
        st.image(image_resized)

    with col2:
        uploaded_file_2 = st.file_uploader("Choose background image")

        if uploaded_file_2 is not None:
            if utils.is_valid_image(uploaded_file_2):
                room_image = utils.process_image(uploaded_file_2)

        st.divider()
        st.write("Background image")
        room_image_resized = utils.resize_image(room_image, image_size_percent2)
        st.image(room_image_resized)

    st.divider()

    code = """
    
    """

    # Source points
    st.write("Choose source points")

    col_1, col_2 = st.columns(2)

    img_width = image_resized.shape[1]
    img_height = image_resized.shape[0]

    with col_1:
        tx_pnts = st.slider("Top x points", 0, img_width, (10, img_width - 10))
        bx_pnts = st.slider("Bot x points", 0, img_width, (10, img_width - 10))

    with col_2:
        ly_pnts = st.slider("Left y points", 0, img_height, (10, img_height - 10))
        ry_pnts = st.slider("Right y points", 0, img_height, (10, img_height - 10))

    point_1 = (tx_pnts[0], ly_pnts[0])
    point_2 = (tx_pnts[1], ry_pnts[0])
    point_3 = (bx_pnts[0], ly_pnts[1])
    point_4 = (bx_pnts[1], ry_pnts[1])

    img_with_dots = utils.draw_dots_on_image(
        image_resized, [point_1, point_2, point_3, point_4]
    )

    st.image(img_with_dots)
    st.divider()

    # Destination points
    st.write("Choose destination points")

    cols_1, cols_2 = st.columns(2)

    img_room_width = room_image_resized.shape[1]
    img_room_height = room_image_resized.shape[0]

    with cols_1:
        txd_pnts = st.slider("Top destination x points", 0, img_room_width, (160, 212))

        bxd_pnts = st.slider(
            "Bot destination x points",
            0,
            img_room_width,
            (160, 212),
        )

    with cols_2:
        lyd_pnts = st.slider("Left destination y points", 0, img_room_height, (60, 130))

        ryd_pnts = st.slider(
            "Right destination y points", 0, img_room_height, (60, 130)
        )

    point_d1 = (txd_pnts[0], lyd_pnts[0])
    point_d2 = (txd_pnts[1], ryd_pnts[0])
    point_d3 = (bxd_pnts[0], lyd_pnts[1])
    point_d4 = (bxd_pnts[1], ryd_pnts[1])

    img_room_with_dots = utils.draw_dots_on_image(
        room_image_resized, [point_d1, point_d2, point_d3, point_d4]
    )

    st.image(img_room_with_dots)
    st.divider()

    # Final combined image
    st.write("Final image")
    src_points = np.float32([[point_1], [point_2], [point_3], [point_4]])
    dst_points = np.float32([[point_d1], [point_d2], [point_d3], [point_d4]])

    final_img = homography_transformation(
        image_resized, room_image_resized, src_points, dst_points
    )
    st.image(final_img)

    # Provide the download link
    st.download_button(
        label="Download Result Image",
        data=utils.prepare_image_for_download(final_img),
        key="download_button",
        file_name="final_image.jpg",
    )
