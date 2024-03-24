import streamlit as st
from . import utils
import numpy as np
import cv2


def sidebar():
    st.sidebar.divider()
    st.sidebar.write("Debug console")

    st.sidebar.divider()
    st.sidebar.write("Resize image so it fits good on your screen")
    image_size_percent = st.sidebar.slider("Image size in percents", 0, 100, 50)

    return image_size_percent


def scaling_matrix(image, scale_x, scale_y):
    scaling_matrix = np.array(
        [[scale_x, 0, 0], [0, scale_y, 0], [0, 0, 1]], dtype=np.float32
    )

    scaled_image = cv2.warpPerspective(
        image.copy(),
        scaling_matrix,
        (int(image.shape[1] * scale_x), int(image.shape[0] * scale_y)),
    )

    return scaled_image


def skew_matrix(image, skew):
    skew_radians = np.radians(skew)

    skew_matrix = np.array(
        [[1, np.tan(skew_radians), 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32
    )

    original_width, original_height = image.shape[1], image.shape[0]
    new_width = int(
        original_width * np.cos(skew_radians)
        + original_height * np.abs(np.sin(skew_radians))
    )
    new_height = original_height

    skewed_image = cv2.warpPerspective(
        image.copy(), skew_matrix, (new_width, new_height)
    )

    return skewed_image


def translation_matrix(image, tx, ty):
    translation_matrix = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float32)

    translated_image = cv2.warpPerspective(
        image.copy(), translation_matrix, (image.shape[1], image.shape[0])
    )

    return translated_image


def rotation_matrix(image, angle_degrees):
    angle_radians = np.radians(angle_degrees)

    center_x = image.shape[1] // 2
    center_y = image.shape[0] // 2

    rotation_matrix = np.array(
        [
            [
                np.cos(angle_radians),
                -np.sin(angle_radians),
                (1 - np.cos(angle_radians)) * center_x
                + np.sin(angle_radians) * center_y,
            ],
            [
                np.sin(angle_radians),
                np.cos(angle_radians),
                -np.sin(angle_radians) * center_x
                + (1 - np.cos(angle_radians)) * center_y,
            ],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )
    rotated_image = cv2.warpPerspective(
        image, rotation_matrix, (image.shape[1], image.shape[0])
    )

    return rotated_image


def combined_matrix(image, tx, ty, angle_degrees, scale_x, scale_y, skew):
    angle_radians = np.radians(angle_degrees)
    translation_matrix = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float32)

    rotation_matrix = np.array(
        [
            [np.cos(angle_radians), -np.sin(angle_radians), 0],
            [np.sin(angle_radians), np.cos(angle_radians), 0],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )

    scaling_matrix = np.array(
        [[scale_x, 0, 0], [0, scale_y, 0], [0, 0, 1]], dtype=np.float32
    )

    skewing_matrix = np.array(
        [[1, np.tan(np.radians(skew)), 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32
    )

    composite_matrix = np.dot(
        skewing_matrix,
        np.dot(scaling_matrix, np.dot(rotation_matrix, translation_matrix)),
    )

    transformed_image = cv2.warpPerspective(
        image.copy(), composite_matrix, (image.shape[1], image.shape[0])
    )

    return transformed_image


def main(image):
    image_size_percent = sidebar()

    st.title("3x3 transformations")

    st.divider()
    st.write("You can upload your own image or use template image")

    uploaded_file = st.file_uploader("Choose your personal image")

    if uploaded_file is not None:
        if utils.is_valid_image(uploaded_file):
            image = utils.process_image(uploaded_file)

    st.divider()

    image_resized = utils.resize_image(image, image_size_percent)
    st.image(image_resized)
    st.divider()

    # ---------- Scaling ----------
    st.title("Scaling matrix")

    code = """
    scaling_matrix = np.array([
            [scale_x, 0, 0],
            [0, scale_y, 0],
            [0, 0, 1]
            ], dtype=np.float32)

    scaled_image = cv2.warpPerspective(image, scaling_matrix, (int(image.shape[1] * scale_x), int(image.shape[0] * scale_y)))
    """

    st.code(code, language="python")

    scale_col_1, scale_col_2 = st.columns(2)

    with scale_col_1:
        scale_x = st.slider("Scele X", 1.0, 3.0, 1.8)

    with scale_col_2:
        scale_y = st.slider("Scele Y", 1.0, 3.0, 1.4)

    scaled_image = scaling_matrix(image_resized, scale_x, scale_y)
    st.image(scaled_image)

    # ---------- Skew ----------
    st.divider()

    st.title("Skew matrix")

    code = """
    skew_radians = np.radians(skew)

    skew_matrix = np.array([
        [1, np.tan(skew_radians), 0],
        [0, 1, 0],
        [0, 0, 1]
        ], dtype=np.float32)2)
    
    original_width, original_height = image.shape[1], image.shape[0]
    new_width = int(original_width * np.cos(skew_radians) + original_height * np.abs(np.sin(skew_radians)))
    new_height = original_height
    
    skewed_image = cv2.warpPerspective(image.copy(), skew_matrix, (new_width, new_height))
    """

    st.code(code, language="python")

    skew = st.slider("Skew", 0, 50, 20)

    skewed_image = skew_matrix(image_resized, skew)
    st.image(skewed_image)

    # ---------- Translation ----------
    st.divider()

    st.title("Translation matrix")

    code = """
    translation_matrix = np.array([
        [1, 0, tx],
        [0, 1, ty],
        [0, 0, 1]
        ], dtype=np.float32)

    translated_image = cv2.warpPerspective(image.copy(), translation_matrix, (image.shape[1], image.shape[0]))
    """

    st.code(code, language="python")

    translat_col_1, translat_col_2 = st.columns(2)
    max_width = image_resized.shape[1] - 5
    max_height = image_resized.shape[0] - 5

    with translat_col_1:
        tx = st.slider("Translation X (tx)", -max_width, max_width, 20)

    with translat_col_2:
        ty = st.slider("Translation Y (ty)", -max_height, max_height, 20)

    translated_image = translation_matrix(image_resized, tx, ty)
    st.image(translated_image)

    # ---------- Rotation ----------
    st.divider()

    st.title("Rotation matrix")

    code = """
    angle_radians = np.radians(angle_degrees)
    center_x = image.shape[1] // 2
    center_y = image.shape[0] // 2

    rotation_matrix = np.array([[np.cos(angle_radians), -np.sin(angle_radians), (1 - np.cos(angle_radians)) * center_x + np.sin(angle_radians) * center_y],
                                [np.sin(angle_radians), np.cos(angle_radians), -np.sin(angle_radians) * center_x + (1 - np.cos(angle_radians)) * center_y],
                                [0, 0, 1]], dtype=np.float32)
    rotated_image = cv2.warpPerspective(image, rotation_matrix, (image.shape[1], image.shape[0]))
    """

    st.code(code, language="python")

    rotation = st.slider("Rotation angle (degrees)", -90, 90, 45)

    rotated_image = rotation_matrix(image_resized, rotation)
    st.image(rotated_image)

    # ---------- Combined matrix ----------
    st.divider()

    st.title("Combined matrices")

    code = """
       composite_matrix = np.dot(skewing_matrix, np.dot(scaling_matrix, np.dot(rotation_matrix, translation_matrix)))
       transformed_image = cv2.warpPerspective(image.copy(), composite_matrix, (image.shape[1], image.shape[0]))
    """

    st.code(code, language="python")

    scalec_col_1, scalec_col_2 = st.columns(2)

    with scalec_col_1:
        scalec_x = st.slider("Scele combined X", 1.0, 3.0, 1.8)

    with scalec_col_2:
        scalec_y = st.slider("Scele combined Y", 1.0, 3.0, 1.4)

    translat_col_1, translat_col_2 = st.columns(2)
    max_width = image_resized.shape[1] - 5
    max_height = image_resized.shape[0] - 5

    with translat_col_1:
        txc = st.slider("Translation X combined (tx)", -max_width, max_width, 0)

    with translat_col_2:
        tyc = st.slider("Translation Y combined (ty)", -max_height, max_height, -50)

    skew_rot_col_1, skew_rot_col_2 = st.columns(2)

    with skew_rot_col_1:
        skewc = st.slider("Skew combined", 0, 50, 20)

    with skew_rot_col_2:
        rotationc = st.slider("Rotation angle combined (degrees)", -90, 90, 45)

    combined_image = combined_matrix(
        image_resized, txc, tyc, rotationc, scalec_x, scalec_y, skewc
    )

    st.image(combined_image)
