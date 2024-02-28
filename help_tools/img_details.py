import streamlit as st
import numpy as np
import cv2


def collect_img_data(image):
    debug_dict = {}

    # Get width, height, and shape of the image
    debug_dict["width"] = image.shape[1]
    debug_dict["height"] = image.shape[0]
    debug_dict["shape"] = image.shape

    # Get channel count
    debug_dict["channel_count"] = image.shape[2] if len(image.shape) == 3 else 1

    channel_info = []

    # Get min and max value for each channel
    if debug_dict["channel_count"] > 1:
        for i in range(debug_dict["channel_count"]):
            channel_min = np.min(image[:, :, i])
            channel_max = np.max(image[:, :, i])
            channel_info.append([channel_min, channel_max])
            # debug_dict[f"min_value_channel_{i+1}"] = channel_min
            # debug_dict[f"max_value_channel_{i+1}"] = channel_max
    else:
        channel_info.append([np.min(image), np.max(image)])
        # debug_dict["min_value"] = np.min(image)
        # debug_dict["max_value"] = np.max(image)

    debug_dict["channel_min_max_value"] = channel_info

    return debug_dict


def main(image):
    st.title("Info about image")

    st.divider()
    st.write("You can upload your own image or use template image")

    uploaded_file = st.file_uploader("Choose your personal image")

    if uploaded_file is not None:
        valid_extensions = ["jpg", "jpeg", "png"]
        file_extension = uploaded_file.name.split(".")[-1].lower()
        if file_extension in valid_extensions:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    img_data = collect_img_data(image)
    st.write(img_data)
    col_a, col_b = st.columns(2)

    with col_a:
        st.image(image)

    with col_b:
        st.write("Info about image:")
        st.write(f"    * Width  = {img_data['width']}")
        st.write(f"    * Height = {img_data['height']}")
        st.write(f"    * Shape = {img_data['shape']}")
        st.write(f"    * Channel count = {img_data['channel_count']}")
        st.write(f"    * Channel min-max values = {img_data['channel_min_max_value']}")

    st.divider()
