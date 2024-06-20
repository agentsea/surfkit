import base64
import io
from io import BytesIO
from typing import List, Tuple

import replicate
import requests
from PIL import Image, ImageDraw, ImageFont


class Box:
    """
    Represents a rectangular box with integer coordinates.

    The `Box` class represents a rectangular area defined by its left, top, right, and bottom coordinates. It provides methods for performing common operations on the box, such as calculating its width and height, zooming in on a specific cell within the box, cropping an image to the box's dimensions, and drawing the box on a drawing context.

    The `Box` class is used throughout the `surfpizza` module to represent and manipulate rectangular areas, such as when processing and displaying images.
    """

    def __init__(self, left: int, top: int, right: int, bottom: int):
        self.left = left
        self.top = top
        self.right = right
        self.bottom = bottom

    def width(self) -> int:
        return self.right - self.left

    def height(self) -> int:
        return self.bottom - self.top

    def zoom_in(self, cell_index: int, num_cells: int) -> "Box":
        cell_width = self.width() // num_cells
        cell_height = self.height() // num_cells
        col = (cell_index - 1) % num_cells
        row = (cell_index - 1) // num_cells
        return Box(
            self.left + col * cell_width,
            self.top + row * cell_height,
            self.left + (col + 1) * cell_width,
            self.top + (row + 1) * cell_height,
        )

    def center(self) -> Tuple[int, int]:
        return ((self.left + self.right) // 2, (self.top + self.bottom) // 2)

    def crop_image(self, img: Image.Image) -> Image.Image:
        return img.crop((self.left, self.top, self.right, self.bottom))

    def draw(
        self,
        draw_context,
        outline: str = "red",
        width: int = 3,
    ) -> None:
        draw_context.rectangle(
            [self.left, self.top, self.right, self.bottom], outline=outline, width=width
        )

    def to_absolute(self, parent_box: "Box") -> "Box":
        return Box(
            self.left + parent_box.left,
            self.top + parent_box.top,
            self.right + parent_box.left,
            self.bottom + parent_box.top,
        )


def divide_image_into_cells(
    image: Image.Image, num_cells: int
) -> Tuple[Image.Image, List[Image.Image], List[Box]]:
    """Divides an image into a grid of cells, returning both the cropped images and their corresponding Box objects.

    Args:
        image (Image.Image): The input image to be divided.
        num_cells (int): The number of cells per row and column.

    Returns:
        Tuple[Image.Image, List[Box]]: A composite image, and a list of boxes corresponding to each cell.
    """
    img_width, img_height = image.size
    cell_width = img_width // num_cells
    cell_height = img_height // num_cells

    cropped_images: List[Image.Image] = []
    boxes: List[Box] = []
    for i in range(num_cells):
        for j in range(num_cells):
            box = Box(
                i * cell_width,
                j * cell_height,
                (i + 1) * cell_width if (i + 1) * cell_width < img_width else img_width,
                (
                    (j + 1) * cell_height
                    if (j + 1) * cell_height < img_height
                    else img_height
                ),
            )
            cropped_image = box.crop_image(image)
            cropped_images.append(cropped_image)
            boxes.append(box)

    composite = combine_images_vertically(cropped_images)

    return composite, cropped_images, boxes


def create_grid_image_by_num_cells(
    image_width: int,
    image_height: int,
    color_circle: str = "red",
    color_text: str = "yellow",
    num_cells: int = 6,
) -> Image.Image:
    """Create the pizza grid image.

    Args:
        image_width (int): Width of the image.
        image_height (int): Height of the image.
        color_circle (str): Color of the circles. Defaults to 'red'
        color_text (str): Color of the text. Defaults to 'yellow'
        num_cells (int): The number of cells in each dimension. Defaults to 6.

    Returns:
        Image.Image: The image grid
    """
    cell_width = image_width // num_cells
    cell_height = image_height // num_cells
    font_size = max(cell_height // 5, 30)
    circle_radius = font_size * 7 // 10

    # Create a blank image with transparent background
    img = Image.new("RGBA", (image_width, image_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Load a font
    font = ImageFont.truetype("fonts/arialbd.ttf", font_size)

    # Set the number of cells in each dimension
    num_cells_x = num_cells - 1
    num_cells_y = num_cells - 1

    # Draw the numbers in the center of each cell
    for i in range(num_cells_x):
        for j in range(num_cells_y):
            number = i * num_cells_y + j + 1
            text = str(number)
            x = (i + 1) * cell_width
            y = (j + 1) * cell_height
            draw.ellipse(
                [
                    x - circle_radius,
                    y - circle_radius,
                    x + circle_radius,
                    y + circle_radius,
                ],
                fill=color_circle,
            )
            offset_x = font_size / 4 if number < 10 else font_size / 2
            draw.text(
                (x - offset_x, y - font_size / 2), text, font=font, fill=color_text
            )

    return img


def create_grid_image_by_size(
    image_width: int,
    image_height: int,
    cell_size: int = 10,
    color_circle: str = "red",
    color_text: str = "yellow",
) -> Image.Image:
    """Create a grid image with numbered cells.

    Args:
        image_width (int): Total width of the image.
        image_height (int): Total height of the image.
        cell_size (int): Width and height of each cell.
        color_circle (str): Color of the circles. Defaults to 'red'
        color_text (str): Color of the text. Defaults to 'yellow'

    Returns:
        Image.Image: The image with a grid.
    """
    num_cells_x = image_width // cell_size
    num_cells_y = image_height // cell_size
    font_size = max(cell_size // 5, 10)
    circle_radius = (
        cell_size // 2 - 2
    )  # Slightly smaller than half the cell for visual appeal

    # Create a blank image
    img = Image.new("RGBA", (image_width, image_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Load a font
    try:
        font = ImageFont.truetype("arialbd.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()
        print("Custom font not found. Using default font.")

    # Draw the grid
    for i in range(num_cells_x):
        for j in range(num_cells_y):
            number = i * num_cells_y + j + 1
            text = str(number)
            x_center = (i + 0.5) * cell_size
            y_center = (j + 0.5) * cell_size

            # Draw circle
            draw.ellipse(
                [
                    x_center - circle_radius,
                    y_center - circle_radius,
                    x_center + circle_radius,
                    y_center + circle_radius,
                ],
                fill=color_circle,
            )

            # Calculate text offset for centering using getbbox()
            bbox = font.getbbox(text)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            draw.text(
                (x_center - text_width / 2, y_center - text_height / 2),
                text,
                font=font,
                fill=color_text,
            )

    return img


def combine_images_vertically(images: List[Image.Image]) -> Image.Image:
    """Combine images vertically and draw a small red circle in the center of each image."""
    padding = 10
    line_height = 2
    total_height = sum(image.height + padding * 2 for image in images) + line_height * (
        len(images) - 1
    )
    max_width = max(image.width for image in images) + 100

    combined_image = Image.new("RGB", (max_width, total_height), "white")
    draw = ImageDraw.Draw(combined_image)

    # Attempt to use a larger font; adjust the path as necessary
    try:
        font = ImageFont.truetype("./font/arial.ttf", 36)
    except IOError:
        font = ImageFont.load_default()
        print("Fallback to default font.")

    y_offset = 0
    for index, image in enumerate(images):
        new_y_offset = y_offset + padding
        combined_image.paste(image, (100, new_y_offset))

        # Draw a small red circle in the center of the image
        circle_radius = 5  # Radius of the circle
        center_x = 100 + image.width // 2
        center_y = new_y_offset + image.height // 2
        # draw.ellipse(
        #     [
        #         (center_x - circle_radius, center_y - circle_radius),
        #         (center_x + circle_radius, center_y + circle_radius),
        #     ],
        #     fill="red",
        # )

        draw.text(
            (20, center_y - 18),
            str(index),
            fill="black",
            font=font,
        )
        y_offset = new_y_offset + image.height + padding
        if index < len(images) - 1:
            draw.line(
                [(0, y_offset), (max_width, y_offset)], fill="black", width=line_height
            )
            y_offset += line_height

    return combined_image


def zoom_in(
    img: Image.Image, box: Box, num_cells: int, selected: int
) -> Tuple[Image.Image, Box]:
    """Zoom in on the selected cell.

    Args:
        img (Image.Image): The image to zoom in
        box (Box): The box to zoom into
        num_cells (int): Number of cells to use.
        selected (int): The selected cell

    Returns:
        Tuple[Image.Image, Box]: Cropped image and asociated box
    """
    new_box = box.zoom_in(selected, num_cells)
    absolute_box = new_box.to_absolute(box)
    cropped_img = new_box.crop_image(img)
    return cropped_img, absolute_box


def superimpose_images(
    base: Image.Image, layer: Image.Image, opacity: float = 1
) -> Image.Image:
    """

    Args:
        base (Image.Image): Base image
        layer (Image.Image): Layered image
        opacity (float): How much opacity the layer should have. Defaults to 1.

    Returns:
        Image.Image: The superimposed image
    """
    # Ensure both images have the same size
    if base.size != layer.size:
        raise ValueError("Images must have the same dimensions.")

    # Convert the images to RGBA mode if they are not already
    base = base.convert("RGBA")
    layer = layer.convert("RGBA")

    # Create a new image with the same size as the input images
    merged_image = Image.new("RGBA", base.size)

    # Convert image1 to grayscale
    base = base.convert("L")

    # Paste image1 onto the merged image
    merged_image.paste(base, (0, 0))

    # Create a new image for image2 with adjusted opacity
    image2_with_opacity = Image.blend(
        Image.new("RGBA", layer.size, (0, 0, 0, 0)), layer, opacity
    )

    # Paste image2 with opacity onto the merged image
    merged_image = Image.alpha_composite(merged_image, image2_with_opacity)

    return merged_image


def image_to_b64(img: Image.Image, image_format="PNG") -> str:
    """Converts a PIL Image to a base64-encoded string with MIME type included.

    Args:
        img (Image.Image): The PIL Image object to convert.
        image_format (str): The format to use when saving the image (e.g., 'PNG', 'JPEG').

    Returns:
        str: A base64-encoded string of the image with MIME type.
    """
    buffer = BytesIO()
    img.save(buffer, format=image_format)
    image_data = buffer.getvalue()
    buffer.close()

    mime_type = f"image/{image_format.lower()}"
    base64_encoded_data = base64.b64encode(image_data).decode("utf-8")
    return f"data:{mime_type};base64,{base64_encoded_data}"


def b64_to_image(base64_str: str) -> Image.Image:
    """Converts a base64 string to a PIL Image object.

    Args:
        base64_str (str): The base64 string, potentially with MIME type as part of a data URI.

    Returns:
        Image.Image: The converted PIL Image object.
    """
    # Strip the MIME type prefix if present
    if "," in base64_str:
        base64_str = base64_str.split(",")[1]

    image_data = base64.b64decode(base64_str)
    image = Image.open(BytesIO(image_data))
    return image


def load_image_base64(filepath: str) -> str:
    # Load the image from the file path
    image = Image.open(filepath)
    buffered = BytesIO()

    # Save image to the buffer
    image_format = image.format if image.format else "PNG"
    image.save(buffered, format=image_format)

    # Encode the image to base64
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    # Prepare the mime type
    mime_type = f"image/{image_format.lower()}"

    # Return base64 string with mime type
    return f"data:{mime_type};base64,{img_str}"


def crop_box_around(
    image: Image.Image, x: int, y: int, padding: int = 50
) -> Image.Image:
    # Calculate the boundaries of the box
    left = x - padding
    top = y - padding
    right = x + padding
    bottom = y + padding

    # Crop the image
    cropped_img = image.crop((left, top, right, bottom))

    return cropped_img


from PIL import Image, ImageDraw, ImageFont


def create_composite_image(image_text_pairs: List[Tuple[str, str]]) -> Image.Image:
    """Combine multiple images with associated text labels vertically into a single image."""
    padding = 10
    text_height = 50  # Additional space for text
    images = [Image.open(image_path) for image_path, _ in image_text_pairs]

    # Calculate total height considering image heights, text heights, and padding
    total_height = sum(img.height + padding * 2 + text_height for img in images)
    max_width = max(img.width for img in images) + 100  # Additional space for text

    # Create a new composite image
    composite_image = Image.new("RGB", (max_width, total_height), "white")
    draw = ImageDraw.Draw(composite_image)

    # Attempt to use a larger font; adjust the path as necessary
    try:
        font = ImageFont.truetype("./arial.ttf", 24)
    except IOError:
        font = ImageFont.load_default()
        print("Fallback to default font.")

    y_offset = 0
    for (image_path, text), image in zip(image_text_pairs, images):
        new_y_offset = y_offset + padding
        # Paste the image at an offset to allow for text on the left
        composite_image.paste(image, (100, new_y_offset))

        # Draw the text label
        text_position_x = 20  # Position the text to the left of the image
        text_position_y = (
            new_y_offset + (image.height + text_height) // 2 - 12
        )  # Center text vertically
        draw.text((text_position_x, text_position_y), text, fill="black", font=font)

        # Update y_offset for the next image
        y_offset = new_y_offset + image.height + padding + text_height

    # Save the new composite image
    return composite_image


def upscale_image(img: Image.Image, scale: int = 4) -> Image.Image:

    # Convert the PIL image to bytes
    image_bytes = io.BytesIO()
    img.save(image_bytes, format="JPEG")
    image_bytes.seek(0)

    # Run the replicate model
    output = replicate.run(
        "nightmareai/real-esrgan:350d32041630ffbe63c8352783a26d94126809164e54085352f8326e53999085",
        input={"image": image_bytes, "scale": scale, "face_enhance": False},
    )
    # Fetch the image
    response = requests.get(output)  # type: ignore
    response.raise_for_status()

    # Open the image with PIL
    image = Image.open(BytesIO(response.content))

    return image


def draw_red_box(
    image: Image.Image, point: Tuple[int, int], padding: int
) -> Image.Image:
    """
    Draw a red box around a point in an image using padding.

    :param image_path: Path to the input image
    :param point: Tuple (x, y) indicating the center of the box
    :param padding: Padding around the point to determine the box size
    """
    # Open the image
    draw = ImageDraw.Draw(image)

    # Calculate the box coordinates using padding
    left = point[0] - padding
    top = point[1] - padding
    right = point[0] + padding
    bottom = point[1] + padding

    # Draw the red box
    draw.rectangle([left, top, right, bottom], outline="red", width=3)

    return image
