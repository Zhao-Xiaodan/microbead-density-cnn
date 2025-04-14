
import os
import cv2
import numpy as np
from pathlib import Path

def crop_useful_regions(input_folder, output_folder, crop_size=(512, 512), overlap=100,
                        min_content_percentage=10):
    """
    Creates crops only from the useful part of microscope images (the illuminated circular region)

    Parameters:
    input_folder: str - path to folder containing original images
    output_folder: str - path to save cropped images
    crop_size: tuple - size of crops (width, height)
    overlap: int - number of pixels to overlap between crops
    min_content_percentage: int - minimum percentage of non-black pixels to keep a crop
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

            # Create a mask for the useful region (the bright circular area)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

            # Apply threshold to identify the bright area
            _, binary = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)

            # Find contours in the binary image
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Create mask from the largest contour (which should be the circular or elliptical region)
            mask = np.zeros_like(gray)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                cv2.drawContours(mask, [largest_contour], 0, 255, -1)
            else:
                # If no contours found, create a circular mask at the center
                center_x, center_y = width // 2, height // 2
                radius = min(width, height) // 2 - 10  # Slightly smaller than half the smaller dimension
                cv2.circle(mask, (center_x, center_y), radius, 255, -1)

            # Save mask for debugging (optional)
            mask_name = f"{os.path.splitext(img_file)[0]}_mask.png"
            mask_path = os.path.join(curr_output_folder, mask_name)
            cv2.imwrite(mask_path, mask)

            # Calculate steps for crops
            step_x = crop_size[0] - overlap
            step_y = crop_size[1] - overlap

            # Generate crops
            crop_number = 0
            for y in range(0, height-crop_size[1]+1, step_y):
                for x in range(0, width-crop_size[0]+1, step_x):
                    # Extract the current region of the mask
                    current_mask = mask[y:y+crop_size[1], x:x+crop_size[0]]

                    # Calculate percentage of useful content in this crop
                    mask_percentage = (np.sum(current_mask) / 255) / (crop_size[0] * crop_size[1]) * 100

                    # Only process crops with enough useful content
                    if mask_percentage >= min_content_percentage:
                        # Extract crop from original image
                        crop = img[y:y+crop_size[1], x:x+crop_size[0]]

                        # Save crop
                        base_name = os.path.splitext(img_file)[0]
                        crop_name = f"{base_name}_crop_{crop_number}.png"
                        crop_path = os.path.join(curr_output_folder, crop_name)
                        cv2.imwrite(crop_path, crop)

                        crop_number += 1

            print(f"Created {crop_number} useful crops from {img_path}")

def main():
    # Set your folders here
    input_folder = "original_images"
    output_folder = "cropped_images"

    # Set crop size and overlap
    crop_size = (512, 512)  # adjust based on your needs
    overlap = 50  # adjust overlap between crops
    min_content_percentage = 50  # minimum percentage of useful content to keep a crop

    crop_useful_regions(input_folder, output_folder, crop_size, overlap, min_content_percentage)

if __name__ == "__main__":
    main()
