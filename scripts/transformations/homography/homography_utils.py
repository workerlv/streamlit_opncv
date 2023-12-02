from io import BytesIO
from PIL import Image
import numpy as np
import cv2


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


def draw_dots_on_image(image, point_1, point_2, point_3, point_4):
    img = image.copy()
    cv2.circle(img, point_1, 5, (255, 0, 0), thickness=cv2.FILLED)
    cv2.circle(img, point_2, 5, (255, 0, 0), thickness=cv2.FILLED)
    cv2.circle(img, point_3, 5, (255, 0, 0), thickness=cv2.FILLED)
    cv2.circle(img, point_4, 5, (255, 0, 0), thickness=cv2.FILLED)

    return img


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


def prepare_image_for_download(image):
    result_image_stream = BytesIO()
    result_image_pil = Image.fromarray(image)
    result_image_pil.save(result_image_stream, format="JPEG")

    return result_image_stream
