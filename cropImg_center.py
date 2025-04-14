
import os
import cv2
import numpy as np
from pathlib import Path

def enhanced_crop_with_grid(input_folder, output_folder, crop_size=(512, 512), overlap=100, visualize=True):
    """
    Crops the center quarter of each image and 4 additional regions for training data.
    Visualizes the crop grid exactly as shown in the reference image.

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

            # Create base filename for crops
            base_name = os.path.splitext(img_file)[0]
            crop_number = 0

            # -------------------------------------------
            # PART 1: Process center quarter of the image
            # -------------------------------------------

            # Calculate center quarter of the image
            quarter_height = height // 2
            quarter_width = width // 2
            start_x = quarter_width // 2
            start_y = quarter_height // 2

            # Draw visualization of the grid structure
            if visualize:
                # Red rectangle for center quarter (outer boundary)
                cv2.rectangle(viz_img,
                             (start_x, start_y),
                             (start_x + quarter_width, start_y + quarter_height),
                             (0, 0, 255), 3)  # Red, thickness 3

                # Draw green grid lines within the center quarter
                # Horizontal grid lines
                for i in range(1, 4):  # 3 internal horizontal lines (4 rows total)
                    y = start_y + i * (quarter_height // 4)
                    cv2.line(viz_img,
                            (start_x, y),
                            (start_x + quarter_width, y),
                            (0, 255, 0), 2)  # Green, thickness 2

                # Vertical grid lines
                for i in range(1, 3):  # 2 internal vertical lines (3 columns total)
                    x = start_x + i * (quarter_width // 3)
                    cv2.line(viz_img,
                            (x, start_y),
                            (x, start_y + quarter_height),
                            (0, 255, 0), 2)  # Green, thickness 2

                # Draw outer green box (matches the outer boundary of center quarter)
                cv2.rectangle(viz_img,
                             (start_x, start_y),
                             (start_x + quarter_width, start_y + quarter_height),
                             (0, 255, 0), 2)  # Green, thickness 2

            # Extract the center quarter
            center_img = img[start_y:start_y+quarter_height, start_x:start_x+quarter_width]

            # Create smaller crops from the center quarter
            center_height, center_width = center_img.shape[:2]

            # Calculate steps for smaller crops
            step_x = crop_size[0] - overlap
            step_y = crop_size[1] - overlap

            # Generate smaller crops from the center quarter
            for y in range(0, center_height-crop_size[1]+1, step_y):
                for x in range(0, center_width-crop_size[0]+1, step_x):
                    # Extract crop
                    crop = center_img[y:y+crop_size[1], x:x+crop_size[0]]

                    # Skip empty or nearly empty crops (optional)
                    if np.sum(crop) < 100:  # Skip very dark crops
                        continue

                    # Save crop
                    crop_name = f"{base_name}_crop_{crop_number}.png"
                    crop_path = os.path.join(curr_output_folder, crop_name)
                    cv2.imwrite(crop_path, crop)

                    # Draw rectangle for this crop on visualization image
                    if visualize:
                        # Green rectangles for center area crops
                        # Convert crop coordinates from center_img to original image coordinates
                        orig_x1 = start_x + x
                        orig_y1 = start_y + y
                        orig_x2 = orig_x1 + crop_size[0]
                        orig_y2 = orig_y1 + crop_size[1]

                        cv2.rectangle(viz_img,
                                     (orig_x1, orig_y1),
                                     (orig_x2, orig_y2),
                                     (0, 255, 0), 2)  # Green, thickness 2

                        # Add crop number text
                        cv2.putText(viz_img, str(crop_number),
                                   (orig_x1 + 5, orig_y1 + 20),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                   (255, 255, 255), 2)

                    crop_number += 1

            # -------------------------------------------
            # PART 2: Add the 4 additional crops
            # -------------------------------------------

            # Calculate center quarter boundaries
            quarter_start_x = start_x
            quarter_end_x = start_x + quarter_width
            quarter_start_y = start_y
            quarter_end_y = start_y + quarter_height

            # Calculate the height of each green box row (assuming equal division)
            green_box_height = quarter_height // 4

            # Calculate positions for the 4 additional crops that match the black boxes in the image
            additional_positions = [
                # Left side, upper crop (level with second row of green boxes)
                (quarter_start_x - crop_size[0], quarter_start_y + green_box_height),

                # Left side, lower crop (directly below the upper left one)
                (quarter_start_x - crop_size[0], quarter_start_y + green_box_height + crop_size[1]),

                # Right side, upper crop (adjacent to green box, level with second row)
                (quarter_end_x, quarter_start_y + green_box_height),

                # Right side, lower crop (directly below the upper right one)
                (quarter_end_x, quarter_start_y + green_box_height + crop_size[1])
            ]

            # Process each additional crop
            for corner_x, corner_y in additional_positions:
                # Make sure crop coordinates are within image bounds
                corner_x = max(0, min(corner_x, width - crop_size[0]))
                corner_y = max(0, min(corner_y, height - crop_size[1]))

                # Extract the crop
                corner_crop = img[corner_y:corner_y+crop_size[1], corner_x:corner_x+crop_size[0]]

                # Skip empty or nearly empty crops (optional)
                if np.sum(corner_crop) < 100:  # Skip very dark crops
                    continue

                # Save the crop
                crop_name = f"{base_name}_additional_crop_{crop_number}.png"
                crop_path = os.path.join(curr_output_folder, crop_name)
                cv2.imwrite(crop_path, corner_crop)

                # Draw rectangle for this crop on visualization image
                if visualize:
                    # Use different color for additional crops - use pure black
                    cv2.rectangle(viz_img,
                                 (corner_x, corner_y),
                                 (corner_x + crop_size[0], corner_y + crop_size[1]),
                                 (0, 0, 0), 3)  # Black, thickness 3

                    # Add crop number text
                    cv2.putText(viz_img, str(crop_number),
                               (corner_x + 5, corner_y + 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                               (255, 255, 255), 2)

                crop_number += 1

            print(f"Created {crop_number} crops (center and additional areas) from {img_path}")

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

    enhanced_crop_with_grid(input_folder, output_folder, crop_size, overlap, visualize)

if __name__ == "__main__":
    main()
