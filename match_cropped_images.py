
import os
import pandas as pd
import re

def match_cropped_images_with_density(cropped_images_path, density_with_filenames_csv, output_csv_path):
    """
    Create a CSV file with cropped image filenames and their corresponding density values.

    Parameters:
    -----------
    cropped_images_path : str
        Path to the folder containing cropped images
    density_with_filenames_csv : str
        Path to the CSV that contains original filenames and density values
    output_csv_path : str
        Path where the new CSV with cropped filenames and densities will be saved
    """
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

if __name__ == "__main__":
    cropped_images_path = "cropped_images"
    density_with_filenames_csv = "beadDensity_with_filenames.csv"
    output_csv_path = "cropped_images_with_density.csv"

    result = match_cropped_images_with_density(cropped_images_path, density_with_filenames_csv, output_csv_path)

    # Print the first few rows to verify
    print("\nFirst few rows of the resulting CSV:")
    print(result.head())
