
import os
import cv2
import numpy as np
from pathlib import Path

def crop_center_grid(input_folder, output_folder, grid_size=4, crop_size=512, visualize=True):
    """
    Creates a grid of crops from the center of each image, excluding the 4 corner cells.

    Parameters:
    input_folder: str - path to folder containing original images
    output_folder: str - path to save cropped images
    grid_size: int - size of the grid (e.g., 4 for a 4x4 grid)
    crop_size: int - size of each crop (width and height)
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

            # Get image dimensions
            height, width = img.shape[:2]

            # Calculate the total grid area size
            total_grid_width = grid_size * crop_size
            total_grid_height = grid_size * crop_size

            # Calculate starting position to center the grid
            start_x = max(0, (width - total_grid_width) // 2)
            start_y = max(0, (height - total_grid_height) // 2)

            # Adjust if the grid would go beyond image boundaries
            if start_x + total_grid_width > width:
                start_x = max(0, width - total_grid_width)
            if start_y + total_grid_height > height:
                start_y = max(0, height - total_grid_height)

            # Make a copy of the original image for visualization
            if visualize:
                viz_img = img.copy()

                # Draw the overall grid boundary in red
                cv2.rectangle(viz_img,
                             (start_x, start_y),
                             (start_x + total_grid_width, start_y + total_grid_height),
                             (0, 0, 255), 3)  # Red, thickness 3

            # Get relative path to maintain folder structure in output
            rel_path = os.path.relpath(root, input_folder)

            # Create corresponding output subfolder
            if rel_path != '.':
                curr_output_folder = os.path.join(output_folder, rel_path)
                os.makedirs(curr_output_folder, exist_ok=True)
            else:
                curr_output_folder = output_folder

            # Create base filename for crops
            base_name = os.path.splitext(img_file)[0]

            # Process each grid cell
            crop_count = 0
            for row in range(grid_size):
                for col in range(grid_size):
                    # Skip the 4 corner cells
                    if (row == 0 and col == 0) or \
                       (row == 0 and col == grid_size - 1) or \
                       (row == grid_size - 1 and col == 0) or \
                       (row == grid_size - 1 and col == grid_size - 1):
                        continue

                    # Calculate crop coordinates
                    x = start_x + col * crop_size
                    y = start_y + row * crop_size

                    # Make sure crop is within image bounds
                    if x + crop_size > width or y + crop_size > height:
                        continue

                    # Extract the crop
                    crop = img[y:y+crop_size, x:x+crop_size]

                    # Skip empty or nearly empty crops (optional)
                    if np.sum(crop) < 100:  # Skip very dark crops
                        continue

                    # Save crop
                    crop_name = f"{base_name}_grid_r{row}_c{col}.png"
                    crop_path = os.path.join(curr_output_folder, crop_name)
                    cv2.imwrite(crop_path, crop)

                    # Draw rectangle for this crop on visualization image
                    if visualize:
                        color = (0, 255, 0)  # Green for normal grid cells
                        cv2.rectangle(viz_img,
                                     (x, y),
                                     (x + crop_size, y + crop_size),
                                     color, 2)

                        # Add grid position text
                        text = f"r{row}c{col}"
                        cv2.putText(viz_img, text,
                                   (x + 5, y + 25),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                   (255, 255, 255), 2)

                    crop_count += 1

            print(f"Created {crop_count} crops from {img_path}")

            # Draw the corner cells that are skipped (in black)
            if visualize:
                corner_positions = [
                    (0, 0),  # Top-left
                    (0, grid_size - 1),  # Top-right
                    (grid_size - 1, 0),  # Bottom-left
                    (grid_size - 1, grid_size - 1)  # Bottom-right
                ]

                for row, col in corner_positions:
                    x = start_x + col * crop_size
                    y = start_y + row * crop_size
                    cv2.rectangle(viz_img,
                                 (x, y),
                                 (x + crop_size, y + crop_size),
                                 (0, 0, 0), 2)  # Black for skipped corners

                # Save visualization image
                viz_name = f"{os.path.splitext(img_file)[0]}_grid_visualization.jpg"
                viz_path = os.path.join(curr_output_folder, viz_name)
                cv2.imwrite(viz_path, viz_img)
                print(f"Created visualization image: {viz_path}")

def main():
    # Set your folders here
    input_folder = "original_images"
    output_folder = "cropped_images"

    # Set grid parameters
    grid_size = 4  # 4x4 grid
    crop_size = 512  # Each crop is 512x512 pixels

    # Set whether to create visualization
    visualize = True

    crop_center_grid(input_folder, output_folder, grid_size, crop_size, visualize)

if __name__ == "__main__":
    main()
