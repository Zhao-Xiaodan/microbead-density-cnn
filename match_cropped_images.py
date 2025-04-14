
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
        base_name = row['filename']
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
        # Remove extension for processing and store filename without extension
        cropped_base = os.path.splitext(cropped_file)[0]

        # Extract the original filename part (before _crop)
        match = re.match(r'(.*?)_crop', cropped_base)
        if match:
            original_base = match.group(1)

            # Find corresponding density
            density = None
            for orig_name in filename_to_density:
                # Check if the original name appears in the cropped filename
                if original_base == orig_name or original_base.strip() == orig_name.strip():
                    density = filename_to_density[orig_name]
                    break

            # If no exact match, try fuzzy matching
            if density is None:
                for orig_name in filename_to_density:
                    # Check if the original name's date-time pattern appears in the cropped filename
                    date_time_pattern = re.search(r'(\d{2}-\d{2}-\d{2}.*?\d+\s+\d+\s+\d+)', orig_name)
                    if date_time_pattern and date_time_pattern.group(1) in cropped_base:
                        density = filename_to_density[orig_name]
                        break

            # Add to results with filename without extension
            results.append({'filename': cropped_base, 'density': density})
        else:
            # If no _crop pattern found, add with unknown density
            results.append({'filename': cropped_base, 'density': None})

    # Create DataFrame from results
    result_df = pd.DataFrame(results)

    # Sort the DataFrame
    # First by timestamp in the filename (extract date and time)
    def extract_sort_key(filename):
        # Try to extract time info for sorting
        match = re.search(r'(\d{2}-\d{2}-\d{2}.*?\d+\s+\d+\s+\d+)', filename)
        if match:
            return match.group(1)

        # If no match, try to extract crop number for secondary sorting
        crop_match = re.search(r'_crop_(\d+)', filename)
        if crop_match:
            return crop_match.group(1)

        return filename

    # Sort by original timestamp, then by crop number
    result_df['sort_key'] = result_df['filename'].apply(extract_sort_key)
    result_df.sort_values(['sort_key', 'filename'], inplace=True)
    result_df.drop('sort_key', axis=1, inplace=True)

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
