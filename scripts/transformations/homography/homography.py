from . import homography_utils as HU
import streamlit as st
import numpy as np


def main(image, room_image):
    st.title("Homography")

    st.divider()
    st.write("You can upload your own image or use template image")

    uploaded_file = st.file_uploader("Choose your foreground image")

    if uploaded_file is not None:
        if HU.is_valid_image(uploaded_file):
            image = HU.process_image(uploaded_file)

    uploaded_file_2 = st.file_uploader("Choose background image")

    if uploaded_file_2 is not None:
        if HU.is_valid_image(uploaded_file_2):
            room_image = HU.process_image(uploaded_file_2)

    st.divider()
    st.write("Resize image so it fits good on your screen")

    image_size_percent = st.slider("Image size in percents", 0, 100, 10)

    col1, col2 = st.columns(2)

    with col1:
        st.write("Foreground image")
        image_resized = HU.resize_image(image, image_size_percent)
        st.image(image_resized)

    with col2:
        st.write("Background image")
        room_image_resized = HU.resize_image(room_image, image_size_percent)
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
        bx_pnts = st.slider("Bot x points", 0, img_height, (10, img_width - 10))

    with col_2:
        ly_pnts = st.slider("Left y points", 0, img_height, (10, img_height - 10))
        ry_pnts = st.slider("Right y points", 0, img_height, (10, img_height - 10))

    point_1 = (tx_pnts[0], ly_pnts[0])
    point_2 = (tx_pnts[1], ry_pnts[0])
    point_3 = (bx_pnts[0], ly_pnts[1])
    point_4 = (bx_pnts[1], ry_pnts[1])

    img_with_dots = HU.draw_dots_on_image(
        image_resized, point_1, point_2, point_3, point_4
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

    img_room_with_dots = HU.draw_dots_on_image(
        room_image_resized, point_d1, point_d2, point_d3, point_d4
    )

    st.image(img_room_with_dots)
    st.divider()

    # Final combined image
    st.write("Final image")
    src_points = np.float32([[point_1], [point_2], [point_3], [point_4]])
    dst_points = np.float32([[point_d1], [point_d2], [point_d3], [point_d4]])

    final_img = HU.homography_transformation(
        image_resized, room_image_resized, src_points, dst_points
    )
    st.image(final_img)

    # Provide the download link
    st.download_button(
        label="Download Result Image",
        data=HU.prepare_image_for_download(final_img),
        key="download_button",
        file_name="final_image.jpg",
    )
