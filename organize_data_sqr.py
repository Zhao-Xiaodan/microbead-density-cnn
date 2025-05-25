
#!/usr/bin/env python
import os
import shutil
import pandas as pd
import re
import cv2
import numpy as np
import argparse
from pathlib import Path

def extract_dilution_factor_robust(name):
    """
    Robust dilution factor extraction that handles various naming patterns

    Args:
        name: folder name or filename

    Returns:
        dilution factor string or None if not found
    """
    # Remove common prefixes that might interfere (like [01])
    name_clean = re.sub(r'^\[[^\]]*\]', '', name)  # Remove [01], [02], etc.
    name_clean = re.sub(r'^[^a-zA-Z0-9]*', '', name_clean)  # Remove leading special chars

    # Pattern 1: Standard dilution factor like "10x", "80X", "1280x"
    match = re.search(r'(\d+)[xX](?:[^a-zA-Z0-9]|$)', name_clean)
    if match:
        return match.group(1) + 'X'  # Normalize to uppercase X

    # Pattern 2: Just numbers that might be dilution factors (in folder context)
    match = re.match(r'^(\d+)$', name_clean)
    if match and int(match.group(1)) in [10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120, 10240]:
        return match.group(1) + 'X'

    return None

def organize_images(input_dir=None, output_dir=None, density_csv=None):
    """
    Organize images into a single folder and create a CSV with density mappings.

    **ROBUST VERSION** - Handles paths with special characters like [01], brackets, etc.
    """
    # Get input directory first
    if input_dir is None:
        print("Enter the folder name containing the microscopy images:")
        input_dir = input().strip()

    # Normalize input directory path and validate it exists
    input_dir = os.path.normpath(os.path.abspath(input_dir))

    if not os.path.isdir(input_dir):
        print(f"Error: The input folder '{input_dir}' does not exist.")
        return pd.DataFrame()

    print(f"Input directory: {input_dir}")

    # Set up output paths - save to parent directory of input_dir
    input_parent = os.path.dirname(input_dir)

    if output_dir is None:
        # Save to parent directory of input_dir (e.g., dataset3/ not dataset3/train_images/)
        dataset_dir = input_parent
        images_folder = os.path.join(input_parent, 'original_images')
    else:
        images_folder = output_dir
        dataset_dir = os.path.dirname(images_folder)

    os.makedirs(images_folder, exist_ok=True)
    print(f"Output directory: {images_folder}")

    # Load density CSV - look for it in the same directory as input_dir first
    if density_csv is None:
        # First try: same directory as input_dir
        predicted_densities_path = os.path.join(input_parent, 'predicted_densities.csv')
        if not os.path.exists(predicted_densities_path):
            # Second try: in the input_dir itself
            predicted_densities_path = os.path.join(input_dir, 'predicted_densities.csv')
            if not os.path.exists(predicted_densities_path):
                # Third try: current working directory
                predicted_densities_path = os.path.join(os.getcwd(), 'predicted_densities.csv')
    else:
        predicted_densities_path = density_csv

    print(f"Looking for density CSV at: {predicted_densities_path}")

    density_dict = {}
    try:
        if not os.path.exists(predicted_densities_path):
            raise FileNotFoundError(f"Density CSV not found at {predicted_densities_path}")

        predicted_densities_df = pd.read_csv(predicted_densities_path)
        print(f"✓ Found density CSV with columns: {predicted_densities_df.columns.tolist()}")

        # Validate required columns
        if 'Dilution factor' not in predicted_densities_df.columns:
            raise ValueError("Missing 'Dilution factor' column in CSV")
        if 'Predicted_Density' not in predicted_densities_df.columns:
            raise ValueError("Missing 'Predicted_Density' column in CSV")

        # Create density mapping with multiple variants for robustness
        for _, row in predicted_densities_df.iterrows():
            dilution_factor = str(row['Dilution factor']).strip()
            density_value = row['Predicted_Density']

            # Add multiple variants
            density_dict[dilution_factor.upper()] = density_value
            density_dict[dilution_factor.lower()] = density_value
            # Also add without X suffix for numeric matching
            numeric_part = re.search(r'(\d+)', dilution_factor)
            if numeric_part:
                density_dict[numeric_part.group(1)] = density_value

        print(f"✓ Loaded {len(predicted_densities_df)} density mappings")

    except Exception as e:
        print(f"ERROR: Could not load density CSV: {e}")
        print("This script requires a density CSV file to work properly.")
        print("Please provide the path to predicted_densities.csv using --density_csv argument")
        print("Or place the file in one of these locations:")
        print(f"  1. {os.path.join(input_parent, 'predicted_densities.csv')}")
        print(f"  2. {os.path.join(input_dir, 'predicted_densities.csv')}")
        print(f"  3. {os.path.join(os.getcwd(), 'predicted_densities.csv')}")
        return pd.DataFrame()

    # Initialize lists for the CSV
    folder_filename = []
    filename = []
    predicted_density = []

    image_extensions = ('.tif', '.tiff', '.jpg', '.jpeg', '.png', '.TIF', '.TIFF', '.JPG', '.JPEG', '.PNG')
    print(f"Processing input directory: {input_dir}")

    processed_any_images = False

    # Use os.walk() to handle special characters in paths - but ONLY process the specified input_dir
    for root, dirs, files in os.walk(input_dir):
        # Calculate depth from input directory
        rel_path = os.path.relpath(root, input_dir)

        # Only process direct subdirectories and root - this ensures we only look in the specified input_dir
        if rel_path != '.' and os.sep in rel_path:
            continue

        if rel_path == '.':
            folder_name = os.path.basename(input_dir)
            print(f"Checking root folder: {folder_name}")
        else:
            folder_name = os.path.basename(root)
            print(f"Checking subfolder: {folder_name}")

        # Check if this is a dilution factor folder
        dilution_factor = extract_dilution_factor_robust(folder_name)

        # Only process if it's a dilution factor folder or if there are images in root and no subfolders
        should_process = False
        if dilution_factor:
            should_process = True
        elif rel_path == '.':
            # Check if root has images and no dilution subfolders
            image_files_in_root = [f for f in files if f.lower().endswith(tuple(ext.lower() for ext in image_extensions))]
            dilution_subfolders = [d for d in dirs if extract_dilution_factor_robust(d)]

            if image_files_in_root and not dilution_subfolders:
                should_process = True
                print(f"  Root folder has {len(image_files_in_root)} images and no dilution subfolders")

        if should_process:
            if dilution_factor:
                print(f"Processing dilution folder: {folder_name} -> {dilution_factor}")
            else:
                print(f"Processing root folder: {folder_name}")

            # Get density value
            density_value = None
            if dilution_factor:
                for variant in [dilution_factor, dilution_factor.upper(), dilution_factor.lower()]:
                    if variant in density_dict:
                        density_value = density_dict[variant]
                        break

                if density_value is None:
                    print(f"  Warning: No density found for dilution factor: {dilution_factor}")
                    print(f"  Available factors: {list(set(density_dict.keys()))}")
                else:
                    print(f"  Found density value: {density_value}")

            # Process image files in this directory
            image_files = [f for f in files if f.lower().endswith(tuple(ext.lower() for ext in image_extensions))]
            print(f"  Found {len(image_files)} images")

            for file in image_files:
                try:
                    image_path = os.path.join(root, file)

                    # Create new filename
                    if rel_path == '.' and not dilution_factor:
                        new_filename = file
                    else:
                        new_filename = f"{folder_name}_{file}"

                    # Handle duplicates
                    dst_path = os.path.join(images_folder, new_filename)
                    counter = 1
                    base_name, ext = os.path.splitext(new_filename)
                    while os.path.exists(dst_path):
                        new_filename = f"{base_name}_{counter:03d}{ext}"
                        dst_path = os.path.join(images_folder, new_filename)
                        counter += 1

                    # Copy file
                    shutil.copy2(image_path, dst_path)
                    print(f"    Copied: {file} -> {new_filename}")

                    # Add to lists
                    folder_filename.append(new_filename)
                    filename.append(file)
                    predicted_density.append(density_value)

                    processed_any_images = True

                except Exception as e:
                    print(f"    Error processing {file}: {e}")
        else:
            if rel_path == '.':
                print(f"  Skipping root folder (no images or has dilution subfolders)")
            else:
                print(f"  Skipping folder: {folder_name} (not a dilution factor folder)")

    if not processed_any_images:
        print("No images were processed. Please check your directory structure.")
        print("Expected structure:")
        print(f"  {input_dir}/")
        print("    ├── 10x/ (images here)")
        print("    ├── 20x/ (images here)")
        print("    └── ...")
        return pd.DataFrame()

    # Create DataFrame
    df = pd.DataFrame({
        'folderName_filename': folder_filename,
        'filename': filename,
        'predicted_density': predicted_density
    })

    # Sort by dilution factor
    def extract_number_for_sorting(x):
        match = re.search(r'(\d+)', str(x))
        return float(match.group(1)) if match else 0

    df['sort_key'] = df['folderName_filename'].apply(extract_number_for_sorting)
    df = df.sort_values('sort_key').drop('sort_key', axis=1)

    print(f"Successfully processed {len(df)} images")

    # Save CSV
    image_density_csv_path = os.path.join(images_folder, 'image_density_mapping.csv')
    df.to_csv(image_density_csv_path, index=False)
    print(f"CSV file created at: {image_density_csv_path}")

    # Create bead density CSV
    add_filenames_to_density_csv(df, dataset_dir)

    # Show summary
    print("\nDensity assignment summary:")
    if not df.empty:
        density_counts = df['predicted_density'].value_counts().sort_index()
        for density, count in density_counts.items():
            if pd.notna(density):
                print(f"  Density {density}: {count} images")
            else:
                print(f"  No density (None): {count} images")

    return df

def add_filenames_to_density_csv(image_df, dataset_dir):
    """Create a CSV file with filenames and density values."""
    output_csv_path = os.path.join(dataset_dir, 'beadDensity_with_filenames.csv')

    image_df_copy = image_df.copy()
    image_df_copy['predicted_density'] = image_df_copy['predicted_density'].fillna(-1)

    filenames_without_extension = [os.path.splitext(file)[0] for file in image_df_copy['folderName_filename']]

    result_df = pd.DataFrame({
        'filename': filenames_without_extension,
        'density': image_df_copy['predicted_density']
    })

    result_df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
    print(f"Bead density CSV file created at {output_csv_path}")

    return result_df

def crop_fixed_grid(dataset_dir=None, grid_cols=8, grid_rows=4, crop_size=512, visualize=True):
    """
    Creates a grid of uniform crops from images with calculated overlap.
    **ROBUST VERSION** - Uses os.walk() for better path handling.
    """
    if dataset_dir is None:
        root_dir = os.getcwd()
        dataset_dir = os.path.join(root_dir, 'dataset')

    input_folder = os.path.join(dataset_dir, 'original_images')
    output_folder = os.path.join(dataset_dir, 'cropped_images')
    viz_output_folder = os.path.join(dataset_dir, 'grid_images')

    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(viz_output_folder, exist_ok=True)

    print(f"Crop parameters: {grid_cols}x{grid_rows} grid, {crop_size}px crops")

    # Find image files
    image_files = []
    image_extensions = ('.tif', '.tiff', '.jpg', '.jpeg', '.png', '.TIF', '.TIFF', '.JPG', '.JPEG', '.PNG')

    for root, dirs, files in os.walk(input_folder):
        if root == input_folder:  # Only direct files
            for file in files:
                if file.lower().endswith(tuple(ext.lower() for ext in image_extensions)):
                    image_files.append(file)

    print(f"Found {len(image_files)} images to crop")

    for img_file in image_files:
        try:
            img_path = os.path.join(input_folder, img_file)
            img = cv2.imread(img_path)

            if img is None:
                print(f"Could not read: {img_file}")
                continue

            height, width = img.shape[:2]
            print(f"Processing {img_file}: {width}x{height}")

            # Calculate grid positioning
            h_total = grid_cols * crop_size
            v_total = grid_rows * crop_size

            if h_total > width:
                h_overlap = int(np.ceil((h_total - width) / (grid_cols - 1)))
                h_start = 0
            else:
                h_overlap = 0
                h_start = (width - h_total) // 2

            if v_total > height:
                v_overlap = int(np.ceil((v_total - height) / (grid_rows - 1)))
                v_start = 0
            else:
                v_overlap = 0
                v_start = (height - v_total) // 2

            # Generate coordinates
            x_positions = []
            for col in range(grid_cols):
                if h_overlap > 0:
                    x = col * (crop_size - h_overlap)
                else:
                    x = h_start + col * crop_size
                x_positions.append(max(0, min(x, width - crop_size)))

            y_positions = []
            for row in range(grid_rows):
                if v_overlap > 0:
                    y = row * (crop_size - v_overlap)
                else:
                    y = v_start + row * crop_size
                y_positions.append(max(0, min(y, height - crop_size)))

            # Create visualization
            if visualize:
                viz_img = img.copy()

            # Process grid cells
            base_name = os.path.splitext(img_file)[0]
            crop_count = 0

            for row in range(grid_rows):
                for col in range(grid_cols):
                    x, y = x_positions[col], y_positions[row]

                    # Extract crop
                    crop = img[y:y+crop_size, x:x+crop_size]

                    if crop.shape[0] != crop_size or crop.shape[1] != crop_size:
                        continue

                    if np.mean(crop) < 5:  # Skip very dark crops
                        continue

                    # Save crop
                    crop_name = f"{base_name}_grid_r{row}_c{col}.png"
                    crop_path = os.path.join(output_folder, crop_name)

                    if cv2.imwrite(crop_path, crop):
                        crop_count += 1

                        # Add to visualization
                        if visualize:
                            cv2.rectangle(viz_img, (x, y), (x + crop_size, y + crop_size), (0, 255, 0), 2)
                            cv2.putText(viz_img, f"r{row}c{col}", (x + 5, y + 25),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            print(f"  Created {crop_count} crops")

            # Save visualization
            if visualize:
                viz_name = f"{base_name}_grid_visualization.jpg"
                viz_path = os.path.join(viz_output_folder, viz_name)
                cv2.imwrite(viz_path, viz_img)

        except Exception as e:
            print(f"Error processing {img_file}: {e}")

def match_cropped_images_with_density(dataset_dir):
    """Create CSV with cropped image filenames and density values."""
    cropped_images_path = os.path.join(dataset_dir, 'cropped_images')
    density_with_filenames_csv = os.path.join(dataset_dir, 'beadDensity_with_filenames.csv')
    output_csv_path = os.path.join(dataset_dir, 'cropped_images_with_density.csv')

    os.makedirs(cropped_images_path, exist_ok=True)

    # Read density CSV
    try:
        original_df = pd.read_csv(density_with_filenames_csv)
        print(f"Loaded {len(original_df)} density entries")
    except Exception as e:
        print(f"Error reading density CSV: {e}")
        original_df = pd.DataFrame(columns=['filename', 'density'])

    # Create filename to density mapping
    filename_to_density = {}
    for _, row in original_df.iterrows():
        try:
            if pd.notna(row['filename']):
                base_name = os.path.splitext(str(row['filename']))[0]
                density = row['density'] if pd.notna(row['density']) else None
                filename_to_density[base_name] = density
        except Exception:
            continue

    print(f"Created mapping with {len(filename_to_density)} entries")

    # Find cropped files
    cropped_files = []
    for root, dirs, files in os.walk(cropped_images_path):
        if root == cropped_images_path:
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff')):
                    cropped_files.append(file)

    print(f"Found {len(cropped_files)} cropped images")

    # Match cropped files with densities
    results = []
    matched_count = 0

    for cropped_file in cropped_files:
        cropped_base = os.path.splitext(cropped_file)[0]

        # Extract original filename
        match = re.match(r'(.*?)_grid_r\d+_c\d+', cropped_base)
        density = None

        if match:
            original_base = match.group(1)

            # Try direct match first
            if original_base in filename_to_density:
                density = filename_to_density[original_base]
                matched_count += 1
            else:
                # Try fuzzy matching
                for orig_name in filename_to_density:
                    if orig_name in original_base or original_base in orig_name:
                        density = filename_to_density[orig_name]
                        matched_count += 1
                        break

        results.append({'filename': cropped_base, 'density': density})

    print(f"Matched density for {matched_count}/{len(cropped_files)} crops")

    # Create and save results
    if results:
        result_df = pd.DataFrame(results)

        # Sort by grid position
        def safe_extract_row(filename):
            match = re.search(r'_grid_r(\d+)_c\d+', filename)
            return int(match.group(1)) if match else 999

        def safe_extract_col(filename):
            match = re.search(r'_grid_r\d+_c(\d+)', filename)
            return int(match.group(1)) if match else 999

        def safe_extract_base(filename):
            match = re.match(r'(.*?)_grid_r\d+_c\d+', filename)
            return match.group(1) if match else filename

        result_df['row'] = result_df['filename'].apply(safe_extract_row)
        result_df['col'] = result_df['filename'].apply(safe_extract_col)
        result_df['base_image'] = result_df['filename'].apply(safe_extract_base)

        result_df = result_df.sort_values(['base_image', 'row', 'col'])
        result_df = result_df.drop(['row', 'col', 'base_image'], axis=1)

        # Save only the main CSV file - no duplicate density.csv
        result_df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
        print(f"Results saved to {output_csv_path}")

        return result_df
    else:
        return pd.DataFrame()

def main():
    """Main function to run the entire workflow."""
    parser = argparse.ArgumentParser(description='Process microscopy images for CNN training')
    parser.add_argument('--input_dir', type=str, default=None,
                      help='Directory containing input images')
    parser.add_argument('--output_dir', type=str, default=None,
                      help='Directory to save processed images')
    parser.add_argument('--density_csv', type=str, default=None,
                      help='CSV file with density values')
    parser.add_argument('--grid_cols', type=int, default=8,
                      help='Number of grid columns (default: 8)')
    parser.add_argument('--grid_rows', type=int, default=4,
                      help='Number of grid rows (default: 4)')
    parser.add_argument('--crop_size', type=int, default=512,
                      help='Size of each crop in pixels (default: 512)')
    parser.add_argument('--no_visualize', action='store_false', dest='visualize',
                      help='Disable visualization image creation')
    parser.add_argument('--skip_organize', action='store_true',
                      help='Skip organizing images step')
    args = parser.parse_args()

    # Determine the correct dataset directory - save to parent of input_dir
    if args.input_dir:
        input_dir = os.path.normpath(os.path.abspath(args.input_dir))
        dataset_dir = os.path.dirname(input_dir)  # Save to parent directory of input_dir
    else:
        # Fallback to current directory if no input_dir specified
        dataset_dir = os.getcwd()

    print(f"Using dataset directory: {dataset_dir}")

    if not args.skip_organize:
        print("\n== Step 1: Organizing Images ==")
        result_df = organize_images(args.input_dir, args.output_dir, args.density_csv)

        # If organize failed, don't continue
        if result_df.empty:
            print("Organization failed. Cannot continue with cropping.")
            return

    print("\n== Step 2: Cropping Images ==")
    crop_fixed_grid(dataset_dir, args.grid_cols, args.grid_rows, args.crop_size, args.visualize)

    print("\n== Step 3: Matching Cropped Images with Density Values ==")
    match_cropped_images_with_density(dataset_dir)

    print("\nWorkflow completed successfully!")
    print(f"All outputs saved in: {dataset_dir}")

if __name__ == "__main__":
    main()
