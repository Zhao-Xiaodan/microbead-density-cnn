
import os
import cv2
import numpy as np
from pathlib import Path

def crop_center_quarter_then_subdivide(input_folder, output_folder, crop_size=(512, 512), overlap=100, visualize=True):
    """
    First crops the center quarter of each image, then subdivides that center portion into smaller crops with overlap.
    Can also visualize the crops on the original image.

    Parameters:
    input_folder: str - path to folder containing original images
    output_folder: str - path to save cropped images
    crop_size: tuple - size of final crops (width, height)
    overlap: int - number of pixels to overlap between final crops
    visualize: bool - whether to create visualization of crops on original image
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

            # Make a copy of the original image for visualization
            if visualize:
                viz_img = img.copy()

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

            # Step 1: Calculate center quarter of the image
            quarter_height = height // 2
            quarter_width = width // 2
            start_x = quarter_width // 2
            start_y = quarter_height // 2

            # Draw rectangle for center quarter on visualization image
            if visualize:
                # Red rectangle for center quarter
                cv2.rectangle(viz_img,
                             (start_x, start_y),
                             (start_x + quarter_width, start_y + quarter_height),
                             (0, 0, 255), 3)  # Red, thickness 3

            # Extract the center quarter
            center_img = img[start_y:start_y+quarter_height, start_x:start_x+quarter_width]

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
                    crop_name = f"{base_name}_crop_{crop_number}.png"
                    crop_path = os.path.join(curr_output_folder, crop_name)
                    cv2.imwrite(crop_path, crop)

                    # Draw rectangle for this crop on visualization image
                    if visualize:
                        # Green rectangles for smaller crops
                        # Convert crop coordinates from center_img to original image coordinates
                        orig_x1 = start_x + x
                        orig_y1 = start_y + y
                        orig_x2 = orig_x1 + crop_size[0]
                        orig_y2 = orig_y1 + crop_size[1]

                        cv2.rectangle(viz_img,
                                     (orig_x1, orig_y1),
                                     (orig_x2, orig_y2),
                                     (0, 255, 0), 2)  # Green, thickness 2

                        # Optionally add crop number text
                        cv2.putText(viz_img, str(crop_number),
                                   (orig_x1 + 5, orig_y1 + 20),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                   (255, 255, 255), 2)

                    crop_number += 1

            print(f"Created {crop_number} small crops from the center quarter of {img_path}")

            # Save visualization image
            if visualize:
                viz_name = f"{os.path.splitext(img_file)[0]}_visualization.jpg"
                viz_path = os.path.join(curr_output_folder, viz_name)
                cv2.imwrite(viz_path, viz_img)
                print(f"Created visualization image: {viz_path}")

def main():
    # Set your folders here
    input_folder = "original_images"
    output_folder = "cropped_images"

    # Set crop size and overlap
    crop_size = (512, 512)  # adjust based on your needs
    overlap = 50  # adjust overlap between crops

    # Set whether to create visualization
    visualize = True

    crop_center_quarter_then_subdivide(input_folder, output_folder, crop_size, overlap, visualize)

if __name__ == "__main__":
    main()
