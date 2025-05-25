
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import blob_dog
import pandas as pd
import argparse

# Utility functions
def show_images(images, titles, rows=1, cols=None):
    # Automatically determine number of columns if not specified
    if cols is None:
        cols = len(images)

    fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 5))

    # Handle the case of single image/axis
    if cols == 1:
        axs = [axs]

    for i, (image, title) in enumerate(zip(images, titles)):
        ax = axs[i]
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax.set_title(title)
        ax.axis('off')

    plt.tight_layout()
    return fig

def barPlot(data, labels, title, output_filename):
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(x, data, color='orange', alpha=0.7)
    ax.set_xlabel('Image Name')
    ax.set_ylabel('Bead Count')
    ax.set_title(f'Bead Count Comparison: {title}')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90)
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close(fig)

def get_bead_count(radius, radius_cutoff, count_value):
    """
    Convert blob radius to bead count based on specified radius cutoff and count value
    """
    if radius >= radius_cutoff:  # Large beads
        return count_value
    else:  # All other sizes
        return 1

def process_image(image_path, save_dir, radius_cutoff, count_value, show_steps=False, export_hist=False):
    sigma_value = 5
    threshold_value = 0.02
    img = cv2.imread(image_path)

    # Given specifications
    width_mm = 6.14  # Sensor width in mm
    height_mm = 4.92  # Sensor height in mm
    magnification = 20  # Magnification factor

    # Calculate effective width and height at 20X magnification
    effective_width_mm = width_mm / magnification
    effective_height_mm = height_mm / magnification

    # Calculate viewing area (convert mm² to μm²)
    total_area_um2 = effective_width_mm * effective_height_mm * 1e6  # Convert mm² to μm²

    # Process with inversion
    inverted_img = cv2.bitwise_not(img)
    blurred_img_invert = cv2.GaussianBlur(cv2.cvtColor(inverted_img, cv2.COLOR_BGR2GRAY), (5, 5), 3)

    # Blob detection (DoG)
    blobs_dog_invert = blob_dog(blurred_img_invert, max_sigma=sigma_value, threshold=threshold_value)

    # Create a copy of the image for visualization
    display_img = img.copy()

    # For histogram of radii
    radii = blobs_dog_invert[:, 2]  # Extract radius values

    # Count beads based on radius
    total_bead_count = 0

    # Color map for visualization - now dynamically adjusted based on count_value
    color_map = {
        1: (0, 255, 255),     # Yellow for regular beads
        count_value: (0, 0, 255)  # Red for large beads
    }

    for blob in blobs_dog_invert:
        y, x, r = blob
        bead_count = get_bead_count(r, radius_cutoff, count_value)
        total_bead_count += bead_count

        # Draw circles with color based on bead count
        cv2.circle(display_img, (int(x), int(y)), int(r), color_map[bead_count], 2)

    # Calculate density (beads per 100μm x 100μm)
    density = round(float((total_bead_count / total_area_um2) * 10000), 1)

    # Create histogram of radii (only if export_hist is True)
    hist_fig = None
    if export_hist:
        hist_fig, hist_ax = plt.subplots(figsize=(8, 6))
        bins = np.arange(0, 10.5, 0.5)  # Bin edges from 0 to 10 in 0.5 increments
        hist_ax.hist(radii, bins=bins, alpha=0.7, color='skyblue', edgecolor='black')
        hist_ax.set_xlabel('Blob Radius (r)')
        hist_ax.set_ylabel('Frequency')
        hist_ax.set_title('Histogram of Blob Radii')
        hist_ax.grid(alpha=0.3)

        # Add vertical line at the radius cutoff
        hist_ax.axvline(x=radius_cutoff, color='r', linestyle='--', alpha=0.7,
                       label=f'r={radius_cutoff} (1→{count_value} beads)')
        hist_ax.legend()

    if show_steps:
        # Modified to show only original and detected beads images
        images = [img, display_img]
        titles = ['Original Image', 'Detected Beads']
        fig = show_images(images, titles)
        return fig, hist_fig, density, total_bead_count, radii

    return None, None, density, total_bead_count, radii

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Image Analysis for Bead Detection')
    parser.add_argument('--hist', action='store_true', help='Enable histogram export (disabled by default)')
    parser.add_argument('--radius-cutoff', type=float, default=3.0,
                        help='Radius cutoff value for counting multiple beads (default: 3.0)')
    parser.add_argument('--count-value', type=int, default=3,
                        help='Number of beads to count for blobs with radius >= cutoff (default: 3)')
    parser.add_argument('--input_dir', type=str, default=None,
                        help='Input directory path (if not specified, will prompt for input)')
    parser.add_argument('--enable-multi-counting', action='store_true',
                        help='Enable counting large beads as multiple beads (disabled by default)')
    args = parser.parse_args()

    export_hist = args.hist
    radius_cutoff = args.radius_cutoff
    count_value = args.count_value
    input_dir = args.input_dir
    enable_multi_counting = args.enable_multi_counting

    # If multi-counting is disabled, override the radius cutoff to ensure all beads count as 1
    if not enable_multi_counting:
        # Set to a very high value that no blob will reach
        radius_cutoff = 99.0

    # If input_dir is not provided, prompt the user
    if input_dir is None:
        print("\nNo input directory specified via command line.")
        use_cwd = input("Use current working directory? (y/n): ").strip().lower()

        if use_cwd == 'y' or use_cwd == 'yes':
            input_dir = os.getcwd()
            print(f"Using current directory: {input_dir}")
        else:
            while True:
                input_dir = input("Please enter the full path to the input directory: ").strip()

                # Handle paths with or without quotes
                if (input_dir.startswith('"') and input_dir.endswith('"')) or \
                   (input_dir.startswith("'") and input_dir.endswith("'")):
                    input_dir = input_dir[1:-1]  # Remove surrounding quotes

                # Handle paths with special characters or spaces
                expanded_path = os.path.expanduser(input_dir)  # Expand ~ if present
                normalized_path = os.path.normpath(expanded_path)  # Normalize path

                if os.path.isdir(normalized_path):
                    input_dir = normalized_path
                    print(f"Input directory set to: {input_dir}")
                    break
                else:
                    print(f"Error: '{input_dir}' is not a valid directory.")
                    print("Tips:")
                    print("- Make sure to use the complete path")
                    print("- For paths with spaces, you can use quotes or type the path as is")
                    print("- Example: /Users/name/folder with spaces")
                    print("Please try again.")

    # Load folder paths, excluding 'output'
    source_folder = input_dir
    print(f"Using source directory: {source_folder}")
    output_folder = os.path.join(source_folder, 'output')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Initialize lists for overall data collection
    all_folder_filenames = []
    all_filenames = []
    all_bead_counts = []

    # Print configuration
    print(f"\nConfiguration: Export Histograms = {export_hist}, Multiple Counting = {enable_multi_counting}")
    if enable_multi_counting:
        print(f"    Radius Cutoff = {radius_cutoff}, Count Value = {count_value}")
    else:
        print("    All beads will be counted as 1 (multiple counting disabled)")

    folder_ls = [entry.name for entry in os.scandir(source_folder)
                if entry.is_dir() and entry.name != 'output'
                and entry.name != 'testing'
                and not entry.name.startswith('.')]
    folder_ls.sort()

    print(f"Source folder: {source_folder}")
    print(f"Subfolders: {folder_ls}")

    # Process each folder
    for folder_name in folder_ls:
        folder_path = os.path.join(source_folder, folder_name)

        # Get Excel labels using the simplified approach
        labels = []
        try:
            label_path_ls = sorted([f for f in os.listdir(folder_path) if f.endswith('.xlsx') and not f.startswith('~$')])
            if label_path_ls:
                df = pd.read_excel(os.path.join(folder_path, label_path_ls[0]), header=None, engine='openpyxl')
                labels = df.iloc[:, 0].dropna().values.tolist()
                print(f"Successfully read labels from {label_path_ls[0]}")
                print("Labels found:", labels)
        except Exception as e:
            print(f"Error reading Excel file: {e}")

        file_names = []
        densities = []
        bead_counts = []
        all_radii = []

        # Get sorted list of jpg and png files
        fig_ls = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg',
                                                                                   '.png', '.tif'))])

        # Check if we have enough labels
        if labels and len(fig_ls) > len(labels):
            print(f"Warning: More images ({len(fig_ls)}) than labels ({len(labels)})")

        for idx, file in enumerate(fig_ls):
            file_name_no_ext = os.path.splitext(file)[0]
            image_path = os.path.join(folder_path, file)
            fig, hist_fig, density, bead_count, radii = process_image(
                image_path, output_folder, radius_cutoff, count_value, show_steps=True, export_hist=export_hist
            )

            file_names.append(file_name_no_ext)
            densities.append(density)
            bead_counts.append(bead_count)
            all_radii.extend(radii)

            # Save figures
            if fig:
                combined_image_path = os.path.join(output_folder, f"{folder_name}_{file_name_no_ext}_analysis.png")
                fig.suptitle(f"Image: {file_name_no_ext}\nBead Count: {bead_count} (Density: {density:.1f} per 100μm²)", fontsize=16)
                fig.savefig(combined_image_path, dpi=300)
                plt.close(fig)

            if hist_fig:
                hist_path = os.path.join(output_folder, f"{folder_name}_{file_name_no_ext}_radius_histogram.png")
                hist_fig.savefig(hist_path, dpi=300, bbox_inches='tight')
                plt.close(hist_fig)

        # Create DataFrame
        data = {
            'Image_Name': file_names,
            'Bead_Count': bead_counts,
            'Bead_Density': densities
        }

        # Add labels if available
        if labels and len(labels) >= len(file_names):
            data['Label'] = labels[:len(file_names)]
        else:
            data['Label'] = file_names

        df = pd.DataFrame(data)

        # Sort the DataFrame by the dilution factor
        df['Sort_Value'] = df['Label'].astype(str).str.extract('(\d+)').astype(float)
        df = df.sort_values('Sort_Value', ascending=True)
        df = df.drop('Sort_Value', axis=1)

        # Save CSV
        csv_filename = os.path.join(output_folder, f'{folder_name}_bead_analysis.csv')
        df.to_csv(csv_filename, index=False)

        # Generate bar plot
        barplot_filename = os.path.join(output_folder, f'{folder_name}_bead_count_barplot.png')
        barPlot(
            df['Bead_Count'].values,
            df['Label'].values,
            folder_name,
            barplot_filename
        )

        # Create overall radius histogram for this folder (only if export_hist is True)
        if export_hist and all_radii:
            folder_hist_fig, folder_hist_ax = plt.subplots(figsize=(10, 6))
            bins = np.arange(0, 10.5, 0.5)
            folder_hist_ax.hist(all_radii, bins=bins, alpha=0.7, color='skyblue', edgecolor='black')
            folder_hist_ax.set_xlabel('Blob Radius (r)')
            folder_hist_ax.set_ylabel('Frequency')
            folder_hist_ax.set_title(f'Overall Histogram of Blob Radii for {folder_name}')
            folder_hist_ax.grid(alpha=0.3)

            # Add vertical line at the radius cutoff
            folder_hist_ax.axvline(x=radius_cutoff, color='r', linestyle='--', alpha=0.7,
                                  label=f'r={radius_cutoff} (1→{count_value} beads)')
            folder_hist_ax.legend()

            folder_hist_path = os.path.join(output_folder, f"{folder_name}_overall_radius_histogram.png")
            folder_hist_fig.savefig(folder_hist_path, dpi=300, bbox_inches='tight')
            plt.close(folder_hist_fig)

        print(f"Data and plots exported for folder: {folder_name}")

        # Collect data for overall CSV
        for idx, filename in enumerate(file_names):
            all_folder_filenames.append(f"{folder_name}_{filename}")
            all_filenames.append(filename)
            all_bead_counts.append(bead_counts[idx])

    # Create and save overall CSV file
    overall_data = {
        'folderName_filename': all_folder_filenames,
        'filename': all_filenames,
        'bead_counts': all_bead_counts
    }
    overall_df = pd.DataFrame(overall_data)
    # Sort the DataFrame by folderName_filename
    overall_df = overall_df.sort_values('folderName_filename', ascending=True)
    overall_csv_path = os.path.join(output_folder, 'overall_bead_analysis.csv')
    overall_df.to_csv(overall_csv_path, index=False)
    print(f"Overall data exported to: {overall_csv_path}")

    # Print summary of parameters used
    print("\nAnalysis completed with the following parameters:")
    print(f"- Input directory: {source_folder}")
    print(f"- Multiple counting: {'Enabled' if enable_multi_counting else 'Disabled'}")
    if enable_multi_counting:
        print(f"- Radius cutoff: {radius_cutoff}")
        print(f"- Count value for large beads: {count_value}")
    print(f"- Histogram export: {'Enabled' if export_hist else 'Disabled'}")

if __name__ == "__main__":
    main()
