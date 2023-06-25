from PIL import Image


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