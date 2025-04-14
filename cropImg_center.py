
import os
import cv2
import numpy as np
from pathlib import Path

def crop_center_quarter_then_subdivide(input_folder, output_folder, crop_size=(512, 512), overlap=100):
    """
    First crops the center quarter of each image, then subdivides that center portion into smaller crops with overlap

    Parameters:
    input_folder: str - path to folder containing original images
    output_folder: str - path to save cropped images
    crop_size: tuple - size of final crops (width, height)
    overlap: int - number of pixels to overlap between final crops
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

            # Step 1: Crop the center quarter of the image
            quarter_height = height // 2
            quarter_width = width // 2
            start_x = quarter_width // 2
            start_y = quarter_height // 2

            # Extract the center quarter
            center_img = img[start_y:start_y+quarter_height, start_x:start_x+quarter_width]

            # Save the center quarter (optional)
            center_img_name = f"{os.path.splitext(img_file)[0]}_center_quarter.png"
            center_img_path = os.path.join(curr_output_folder, center_img_name)
            cv2.imwrite(center_img_path, center_img)
            print(f"Created center quarter crop from {img_path}")

            # Step 2: Create smaller crops from the center quarter
            center_height, center_width = center_img.shape[:2]

            # Calculate steps for smaller crops
            step_x = crop_size[0] - overlap
            step_y = crop_size[1] - overlap

            # Generate smaller crops
            crop_number = 0
            for y in range(0, center_height-crop_size[1]+1, step_y):
                for x in range(0, center_width-crop_size[0]+1, step_x):
                    # Extract crop
                    crop = center_img[y:y+crop_size[1], x:x+crop_size[0]]

                    # Skip empty or nearly empty crops (optional)
                    if np.sum(crop) < 100:  # Skip very dark crops
                        continue

                    # Save crop
                    base_name = os.path.splitext(img_file)[0]
                    crop_name = f"{base_name}_center_crop_{crop_number}.png"
                    crop_path = os.path.join(curr_output_folder, crop_name)
                    cv2.imwrite(crop_path, crop)

                    crop_number += 1

            print(f"Created {crop_number} small crops from the center quarter of {img_path}")

def main():
    # Set your folders here
    input_folder = "original_images"
    output_folder = "cropped_images"

    # Set crop size and overlap
    crop_size = (512, 512)  # adjust based on your needs
    overlap = 50  # adjust overlap between crops

    crop_center_quarter_then_subdivide(input_folder, output_folder, crop_size, overlap)

if __name__ == "__main__":
    main()
