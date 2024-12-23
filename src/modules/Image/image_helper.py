"""Image file helper"""

import io
import os

from PIL import Image


def save_image(image_data: bytes, clear_filename: str, output_path: str) -> None:
    image = Image.open(io.BytesIO(image_data))
    image = image.convert('RGB')  # Convert to RGB to avoid transparency or RGBA issues
    image_path = os.path.join(output_path, clear_filename + " [CO].jpg")
    image.save(image_path, "JPEG")
    crop_image_to_square(image_path)


def crop_image_to_square(image_path):
    image = Image.open(image_path)
    width, height = image.size
    size = min(width, height)

    # Calculate the coordinates for the crop
    left = (width - size) // 2
    top = (height - size) // 2
    right = left + size
    bottom = top + size

    cropped_image = image.crop((left, top, right, bottom))

    # Override the original image with the cropped version
    cropped_image.save(image_path)
