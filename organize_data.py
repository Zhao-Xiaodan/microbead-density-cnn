
import os
import shutil
import pandas as pd
import re
import cv2
import numpy as np
import argparse
from pathlib import Path

def get_dilution_factor(folder_name):
    # Extract dilution factor from folder name
    # This assumes the dilution factor is stored in a consistent way
    # You may need to adjust this logic based on your actual folder naming convention
    if folder_name.endswith('X'):
        return folder_name
    parts = folder_name.split('_')
    if len(parts) > 1:
        return parts[-1]  # Return the last part after underscore
    return folder_name    # Return the whole name if no underscore

def organize_images():
    """
    Organize images into a single folder and create a CSV with density mappings.
    """
    # Set up paths
    root_dir = os.getcwd()
    dataset_dir = os.path.join(root_dir, 'dataset')

    # Output directory
    images_folder = os.path.join(dataset_dir, 'original_images')
    os.makedirs(images_folder, exist_ok=True)

    # Load predicted densities with the updated column name
    predicted_densities_path = os.path.join(dataset_dir, 'predicted_densities.csv')
    predicted_densities_df = pd.read_csv(predicted_densities_path)

    print(f"CSV columns: {predicted_densities_df.columns.tolist()}")

    # Create a dictionary mapping dilution factors to predicted densities
    # Convert all keys to uppercase to match folder names
    density_dict = {k.upper(): v for k, v in zip(
        predicted_densities_df['Dilution factor'],
        predicted_densities_df['Predicted_Density']
    )}

    print(f"Density dictionary: {density_dict}")

    # Ask user for the subfolder to process
    print("Enter the folder name to process (e.g., 2025-04_cellphone):")
    folder_input = input().strip()

    input_images_dir = os.path.join(dataset_dir, folder_input)

    if not os.path.isdir(input_images_dir):
        print(f"Error: The folder '{input_images_dir}' does not exist.")
        return

    # Initialize lists for the new CSV
    folder_filename = []
    filename = []
    predicted_density = []

    # Process each subfolder with dilution factors directly in input directory
    for folder_name in os.listdir(input_images_dir):
        folder_path = os.path.join(input_images_dir, folder_name)

        # Skip if not a directory or if it's a special directory
        if not os.path.isdir(folder_path) or folder_name in [
            '.ipynb_checkpoints', 'original_images'
        ]:
            continue

        # Skip folders that don't look like dilution factors
        if not folder_name.endswith('X'):
            continue

        # The folder name itself is the dilution factor (e.g., "10X")
        dilution_factor = folder_name

        # Skip if we can't find corresponding density
        if dilution_factor not in density_dict:
            print(f"Warning: No density found for dilution factor: {dilution_factor}")
            continue

        print(f"Processing folder: {folder_name} with density: {density_dict[dilution_factor]}")

        # Process each image in the folder
        for file in os.listdir(folder_path):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                # Create the new filename with folder prefix
                new_filename = f"{folder_name}_{file}"

                # Copy the file to the images directory
                src_path = os.path.join(folder_path, file)
                dst_path = os.path.join(images_folder, new_filename)
                shutil.copy2(src_path, dst_path)

                # Add to our lists for the CSV
                folder_filename.append(new_filename)
                filename.append(file)
                predicted_density.append(density_dict[dilution_factor])

    # Create and save the CSV file
    df = pd.DataFrame({
        'folderName_filename': folder_filename,
        'filename': filename,
        'predicted_density': predicted_density
    })

    # Sort by dilution factor
    def extract_number(x):
        # Extract the dilution factor part (folder name)
        dilution_part = x.split('_')[0]
        # Remove 'X' and convert to float for numeric sorting
        try:
            return float(dilution_part.replace('X', ''))
        except ValueError:
            return 0  # Default value if conversion fails

    # Sort by the numeric part of the dilution factor
    if not df.empty:
        df['sort_key'] = df['folderName_filename'].apply(extract_number)
        df = df.sort_values('sort_key')
        df = df.drop('sort_key', axis=1)
    else:
        print("Warning: No images were processed. The DataFrame is empty.")

    # Save the CSV
    image_density_csv_path = os.path.join(images_folder, 'image_density_mapping.csv')
    df.to_csv(image_density_csv_path, index=False)

    print(f"Successfully processed {len(df)} images")
    print(f"Images copied to: {images_folder}")
    print(f"CSV file created at: {image_density_csv_path}")

    # Create the bead density CSV in the expected format
    add_filenames_to_density_csv(df, dataset_dir)

def add_filenames_to_density_csv(image_df, dataset_dir):
    """
    Create a CSV file with filenames and density values.

    Parameters:
    -----------
    image_df : DataFrame
        DataFrame containing image filenames and density values
    dataset_dir : str
        Path to the dataset directory
    """
    # Directory paths
    images_folder_path = os.path.join(dataset_dir, 'original_images')
    output_csv_path = os.path.join(dataset_dir, 'beadDensity_with_filenames.csv')

    # Prepare the result DataFrame
    filenames_without_extension = [os.path.splitext(file)[0] for file in image_df['folderName_filename']]

    result_df = pd.DataFrame({
        'filename': filenames_without_extension,
        'density': image_df['predicted_density']
    })

    # Save the new DataFrame to a CSV file with UTF-8 encoding
    result_df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
    print(f"Bead density CSV file created at {output_csv_path}")

    return result_df

def crop_center_grid(dataset_dir):
    """
    Creates a grid of crops from the center of each image, excluding the 4 corner cells.
    """
    # Set up paths
    input_folder = os.path.join(dataset_dir, 'original_images')
    output_folder = os.path.join(dataset_dir, 'cropped_images')
    viz_output_folder = os.path.join(dataset_dir, 'grid_images')

    # Create output folders
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(viz_output_folder, exist_ok=True)

    # Default parameters
    grid_size = 4
    crop_size = 512
    visualize = True

    # Walk through all subdirectories
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.tif', '.jpeg'))]
    print(f"Processing files in {input_folder}: {len(image_files)} images found")

    for img_file in image_files:
        # Read image
        img_path = os.path.join(input_folder, img_file)
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
                crop_path = os.path.join(output_folder, crop_name)
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

            # Save visualization image to a separate folder
            viz_name = f"{base_name}_grid_visualization.jpg"
            viz_path = os.path.join(viz_output_folder, viz_name)
            cv2.imwrite(viz_path, viz_img)
            print(f"Created visualization image: {viz_path}")

def match_cropped_images_with_density(dataset_dir):
    """
    Create a CSV file with cropped image filenames and their corresponding density values.
    """
    # Set up paths
    cropped_images_path = os.path.join(dataset_dir, 'cropped_images')
    density_with_filenames_csv = os.path.join(dataset_dir, 'beadDensity_with_filenames.csv')
    output_csv_path = os.path.join(dataset_dir, 'cropped_images_with_density.csv')

    # Read the previously created CSV with original filenames and densities
    original_df = pd.read_csv(density_with_filenames_csv)

    # Create a dictionary mapping original filenames to density values
    # Remove file extensions for proper matching
    filename_to_density = {}
    for _, row in original_df.iterrows():
        # Extract the base part of the filename (without extension)
        base_name = os.path.splitext(row['filename'])[0] if '.' in row['filename'] else row['filename']
        density = row['density']
        filename_to_density[base_name] = density

    # Get list of cropped image filenames
    cropped_files = []
    for file in os.listdir(cropped_images_path):
        if file.endswith(('.jpg', '.jpeg', '.png')):
            cropped_files.append(file)

    # Create a list to store results
    results = []

    # Process each cropped image
    for cropped_file in cropped_files:
        # Remove extension for processing
        cropped_base = os.path.splitext(cropped_file)[0]

        # Extract the original filename part (before _grid)
        # For new format like "10X_照片 04-04-25 下午4 52 54_grid_r0_c1"
        match = re.match(r'(.*?)_grid_r\d+_c\d+$', cropped_base)

        if match:
            original_base = match.group(1)

            # Find corresponding density
            density = None

            # First try direct matching with the base name
            if original_base in filename_to_density:
                density = filename_to_density[original_base]
            else:
                # Extract the date-time pattern to match with original filenames
                date_match = re.search(r'(\d{2}-\d{2}-\d{2}[^_]+)', original_base)
                if date_match:
                    date_pattern = date_match.group(1)

                    # Try to find a matching original filename with this date pattern
                    for orig_name in filename_to_density:
                        if date_pattern in orig_name:
                            density = filename_to_density[orig_name]
                            break

            # Add to results
            results.append({'filename': cropped_base, 'density': density})
        else:
            # If no grid pattern found, add with unknown density
            results.append({'filename': cropped_base, 'density': None})

    # Create DataFrame from results
    result_df = pd.DataFrame(results)

    # Sort the DataFrame
    # First by magnification and timestamp in the filename
    def extract_sort_key(filename):
        # Extract magnification (e.g., 10X, 5X)
        mag_match = re.search(r'^(\d+X)', filename)
        mag = mag_match.group(1) if mag_match else ""

        # Try to extract time info for sorting
        date_match = re.search(r'(\d{2}-\d{2}-\d{2}[^_]+)', filename)
        date_time = date_match.group(1) if date_match else ""

        # Extract grid coordinates for secondary sorting
        grid_match = re.search(r'_grid_r(\d+)_c(\d+)', filename)
        r_num = int(grid_match.group(1)) if grid_match else 0
        c_num = int(grid_match.group(2)) if grid_match else 0

        return (mag, date_time, r_num, c_num)

    # Create sort key columns for explicit sorting
    result_df['magnification'] = result_df['filename'].apply(
        lambda x: re.search(r'^(\d+X)', x).group(1) if re.search(r'^(\d+X)', x) else "")

    result_df['datetime'] = result_df['filename'].apply(
        lambda x: re.search(r'(\d{2}-\d{2}-\d{2}[^_]+)', x).group(1) if re.search(r'(\d{2}-\d{2}-\d{2}[^_]+)', x) else "")

    result_df['row_num'] = result_df['filename'].apply(
        lambda x: int(re.search(r'_grid_r(\d+)_c\d+', x).group(1)) if re.search(r'_grid_r(\d+)_c\d+', x) else 999)

    result_df['col_num'] = result_df['filename'].apply(
        lambda x: int(re.search(r'_grid_r\d+_c(\d+)', x).group(1)) if re.search(r'_grid_r\d+_c(\d+)', x) else 999)

    # Sort by magnification, datetime, row number, and column number
    result_df = result_df.sort_values(['magnification', 'datetime', 'row_num', 'col_num'])

    # Drop the sorting columns before saving
    result_df = result_df.drop(['magnification', 'datetime', 'row_num', 'col_num'], axis=1)

    # Save to CSV
    result_df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
    print(f"New CSV file created at {output_csv_path}")

    return result_df

def main():
    """
    Main function to run the entire workflow.
    """
    # Set up paths
    root_dir = os.getcwd()
    dataset_dir = os.path.join(root_dir, 'dataset')

    # Step 1: Organize images and create initial CSV
    organize_images()

    # Step 2: Crop images and create grid visualizations
    crop_center_grid(dataset_dir)

    # Step 3: Match cropped images with density values
    match_cropped_images_with_density(dataset_dir)

    print("\nComplete workflow finished successfully!")

if __name__ == "__main__":
    main()
