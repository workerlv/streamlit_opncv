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


def draw_dots_on_image(image, points):
    img = image.copy()

    for point in points:
        cv2.circle(img, point, 5, (255, 0, 0), thickness=cv2.FILLED)

    return img


def prepare_image_for_download(image):
    result_image_stream = BytesIO()
    result_image_pil = Image.fromarray(image)
    result_image_pil.save(result_image_stream, format="JPEG")

    return result_image_stream
