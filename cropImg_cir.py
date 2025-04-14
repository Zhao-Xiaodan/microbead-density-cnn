
import os
import cv2
import numpy as np
from pathlib import Path

def crop_with_edge_detection(input_folder, output_folder, crop_size=(512, 512), overlap=100,
                           edge_buffer=50, min_content_percentage=70):
    """
    Creates crops from microscope images while avoiding the edge of the circular field of view

    Parameters:
    input_folder: str - path to folder containing original images
    output_folder: str - path to save cropped images
    crop_size: tuple - size of crops (width, height)
    overlap: int - number of pixels to overlap between crops
    edge_buffer: int - buffer distance from detected edge to avoid
    min_content_percentage: int - minimum percentage of useful area to keep a crop
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Walk through all subdirectories
    for root, dirs, files in os.walk(input_folder):
        # Filter image files
        image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.tif', '.jpeg'))]
        print(f"Processing files in {root}: {len(image_files)} images found")

        for img_file in image_files:
            # Read image
            img_path = os.path.join(root, img_file)
            img = cv2.imread(img_path)

            if img is None:
                print(f"Could not read image: {img_path}")
                continue

            # Get relative path to maintain folder structure in output
            rel_path = os.path.relpath(root, input_folder)
            # Create corresponding output subfolder
            if rel_path != '.':
                curr_output_folder = os.path.join(output_folder, rel_path)
                os.makedirs(curr_output_folder, exist_ok=True)
            else:
                curr_output_folder = output_folder

            # Get image dimensions
            height, width = img.shape[:2]

            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

            # Apply threshold to identify the bright area
            _, binary = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)

            # Find edges using Canny edge detection
            edges = cv2.Canny(binary, 50, 150)

            # Dilate edges to create a buffer zone
            kernel = np.ones((edge_buffer, edge_buffer), np.uint8)
            edge_zone = cv2.dilate(edges, kernel, iterations=1)

            # Create mask: 255 for good areas (inside circle, away from edge), 0 for bad areas
            # Invert the edge_zone so edges are 0, non-edges are 255
            safe_zone_mask = cv2.bitwise_not(edge_zone)

            # Also mask out areas outside the circle (dark areas)
            safe_zone_mask = cv2.bitwise_and(safe_zone_mask, binary)

            # Save masks for debugging
            cv2.imwrite(os.path.join(curr_output_folder, f"{os.path.splitext(img_file)[0]}_edges.png"), edges)
            cv2.imwrite(os.path.join(curr_output_folder, f"{os.path.splitext(img_file)[0]}_safe_zone.png"), safe_zone_mask)

            # Calculate steps for crops
            step_x = crop_size[0] - overlap
            step_y = crop_size[1] - overlap

            # Generate crops
            crop_number = 0
            for y in range(0, height-crop_size[1]+1, step_y):
                for x in range(0, width-crop_size[0]+1, step_x):
                    # Extract the current region of the mask
                    current_mask = safe_zone_mask[y:y+crop_size[1], x:x+crop_size[0]]

                    # Calculate percentage of useful content in this crop
                    mask_percentage = (np.sum(current_mask) / 255) / (crop_size[0] * crop_size[1]) * 100

                    # Only process crops with enough useful content and no edge interference
                    if mask_percentage >= min_content_percentage:
                        # Extract crop from original image
                        crop = img[y:y+crop_size[1], x:x+crop_size[0]]

                        # Save crop
                        base_name = os.path.splitext(img_file)[0]
                        crop_name = f"{base_name}_crop_{crop_number}.png"
                        crop_path = os.path.join(curr_output_folder, crop_name)
                        cv2.imwrite(crop_path, crop)

                        crop_number += 1

            print(f"Created {crop_number} edge-free crops from {img_path}")

def main():
    # Set your folders here
    input_folder = "original_images"
    output_folder = "cropped_images"

    # Set parameters
    crop_size = (512, 512)  # adjust based on your needs
    overlap = 50  # adjust overlap between crops
    edge_buffer = 40  # pixels to stay away from detected edges
    min_content_percentage = 90  # minimum percentage of useful content to keep a crop

    crop_with_edge_detection(input_folder, output_folder, crop_size, overlap, edge_buffer, min_content_percentage)

if __name__ == "__main__":
    main()
