
import os
import pandas as pd
import re

def add_filenames_to_density_csv(density_csv_path, images_folder_path, output_csv_path):
    """
    Add image filenames as a column to the bead density CSV file.

    Parameters:
    -----------
    density_csv_path : str
        Path to the CSV file containing the bead density values
    images_folder_path : str
        Path to the folder containing the original images
    output_csv_path : str
        Path where the new CSV with filenames and densities will be saved
    """
    # Read the density CSV file
    density_df = pd.read_csv(density_csv_path)

    # Get list of image filenames from the folder
    image_files = []
    for file in os.listdir(images_folder_path):
        if file.endswith(('.jpg', '.jpeg', '.png')):
            image_files.append(file)

    # Extract timestamps from filenames for sorting
    # The pattern looks for date-time format like "16-10-24...1 28 56.jpg"
    def extract_datetime(filename):
        # Try to extract date and time information using regex
        match = re.search(r'(\d{2}-\d{2}-\d{2}).*?(\d+\s+\d+\s+\d+)', filename)
        if match:
            date, time = match.groups()
            # Convert to a sortable format
            return f"{date} {time.replace(' ', '')}"
        return filename  # Fallback to the original filename

    # Sort image files based on the extracted datetime
    image_files.sort(key=extract_datetime)

    # Check if the number of images matches the number of density values
    if len(image_files) != len(density_df):
        print(f"Warning: Number of images ({len(image_files)}) doesn't match number of density values ({len(density_df)})")

    # Create a new DataFrame with filenames (remove .jpg extension) and density values
    filenames_without_extension = [os.path.splitext(file)[0] for file in image_files[:len(density_df)]]

    result_df = pd.DataFrame({
        'filename': filenames_without_extension,
        '10E6 beads/mL': density_df['10E6 beads/mL']
    })

    # Save the new DataFrame to a CSV file with UTF-8 encoding
    result_df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
    print(f"New CSV file created at {output_csv_path}")

    return result_df

# Example usage
if __name__ == "__main__":
    density_csv_path = "beadDensity.csv"
    images_folder_path = "original_images"
    output_csv_path = "beadDensity_with_filenames.csv"

    result = add_filenames_to_density_csv(density_csv_path, images_folder_path, output_csv_path)

    # Print the first few rows to verify
    print("\nFirst few rows of the resulting CSV:")
    print(result.head())

    print("\nEncoding information used in the output file: UTF-8-sig (with BOM)")
    print("NOTE: File extensions (.jpg) have been removed from filenames")
