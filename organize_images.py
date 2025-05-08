
import os
import shutil
import pandas as pd

def main():
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

    # Process each subfolder with dilution factors directly in dataset directory
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
    csv_path = os.path.join(images_folder, 'image_density_mapping.csv')
    df.to_csv(csv_path, index=False)

    print(f"Successfully processed {len(df)} images")
    print(f"Images copied to: {images_folder}")
    print(f"CSV file created at: {csv_path}")

if __name__ == '__main__':
    main()
