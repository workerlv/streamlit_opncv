import streamlit as st
import numpy as np
import cv2

# ------------ utils ------------


def is_valid_image(file):
    valid_extensions = ["jpg", "jpeg", "png"]
    file_extension = file.name.split(".")[-1].lower()
    return file_extension in valid_extensions


def process_image(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image_rgb


def resize_image(image, percent):
    new_width = int(image.shape[1] * percent / 100)
    new_height = int(image.shape[0] * percent / 100)
    resized_image = cv2.resize(image, (new_width, new_height))

    return resized_image


# ------------ 3x3 transformatons ------------


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
    # Convert degrees to radians
    angle_radians = np.radians(angle_degrees)

    # Calculate the center of the image
    center_x = image.shape[1] // 2
    center_y = image.shape[0] // 2

    # Create individual transformation matrices
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

    # Combine the individual transformation matrices into one composite transformation matrix
    composite_matrix = np.dot(
        skewing_matrix,
        np.dot(scaling_matrix, np.dot(rotation_matrix, translation_matrix)),
    )

    # Apply the composite transformation using cv2.warpPerspective with the 3x3 matrix
    transformed_image = cv2.warpPerspective(
        image.copy(), composite_matrix, (image.shape[1], image.shape[0])
    )

    return transformed_image
