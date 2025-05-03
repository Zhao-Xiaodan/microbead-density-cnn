
import tempfile
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
from PIL import Image
import os
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import argparse
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime
try:
    from natsort import natsorted
    NATSORT_AVAILABLE = True
except ImportError:
    NATSORT_AVAILABLE = False
    print("natsort not available. Install with: pip install natsort")
    print("Using basic natural sorting as fallback.")
    import re

# 1. Define the CNN model - must match the architecture used for training
class DensityRegressionCNN(nn.Module):
    def __init__(self, filters=[128, 256, 512]):
        super(DensityRegressionCNN, self).__init__()
        # Store configuration
        self.filters = filters

        # First convolutional block
        self.conv1 = nn.Conv2d(1, filters[0], kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(filters[0])

        # Second convolutional block
        self.conv2 = nn.Conv2d(filters[0], filters[1], kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(filters[1])

        # Third convolutional block
        self.conv3 = nn.Conv2d(filters[1], filters[2], kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(filters[2])

        # Common layers
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

        # Fully connected layers for regression
        self.fc1 = nn.Linear(filters[2], 64)
        self.fc2 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Convolutional stack with pooling
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))

        # Global Average Pooling
        x = x.mean(dim=[2, 3])

        # Fully connected layers
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# Natural sorting helper function
def natural_sort_key(s):
    """Create a key for natural sorting (e.g., 2X comes before 10X)"""
    # Convert string to list of integers and strings
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', str(s))]

# Natural sorting wrapper - uses natsort if available, fallback otherwise
def natural_sort(items, key_func=None):
    """
    Perform natural sorting on a list of items.
    Uses natsort library if available, otherwise falls back to custom implementation.
    """
    if NATSORT_AVAILABLE:
        if key_func:
            # For complex objects, extract the sorting key first
            keys = [key_func(item) for item in items]
            return [items[i] for i in natsorted(range(len(items)), key=lambda i: keys[i])]
        else:
            # For simple strings/values
            return natsorted(items)
    else:
        # Fallback to custom implementation
        if key_func:
            return sorted(items, key=lambda x: natural_sort_key(key_func(x)))
        else:
            return sorted(items, key=natural_sort_key)

# 2. Function to crop images in a grid pattern
def crop_center_grid(image_path, output_folder, grid_size=4, crop_size=512, visualize=True):
    """
    Creates a grid of crops from the center of the image, excluding the 4 corner cells.
    Returns list of crop paths.
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Read image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not read image: {image_path}")
        return [], [], None

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
    viz_path = None
    if visualize:
        viz_img = img.copy()
        # Draw the overall grid boundary in red
        cv2.rectangle(viz_img,
                     (start_x, start_y),
                     (start_x + total_grid_width, start_y + total_grid_height),
                     (0, 0, 255), 3)  # Red, thickness 3

    # Create base filename for crops
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    # Process each grid cell
    crop_paths = []
    crop_info = []  # Store row, col for each crop
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
            crop_paths.append(crop_path)
            crop_info.append((row, col))

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
        viz_name = f"{base_name}_grid_visualization.jpg"
        viz_path = os.path.join(output_folder, viz_name)
        cv2.imwrite(viz_img, viz_path)
        print(f"    Created visualization image: {viz_path}")

    return crop_paths, crop_info, viz_path

# 3. Function to predict density for a single image
def predict_density(model, image_path, transform, device):
    """
    Predict the microbead density for a single image
    """
    try:
        # Load and preprocess the image
        image = Image.open(image_path).convert('L')  # Convert to grayscale
        image_tensor = transform(image).unsqueeze(0).to(device)

        # Forward pass
        model.eval()
        with torch.no_grad():
            prediction = model(image_tensor)

        return prediction.item()

    except Exception as e:
        print(f"Error predicting for image {image_path}: {str(e)}")
        return None
    # 4. NEW DIRECT BOXPLOT FUNCTION - Creates boxplots with exact Excel order
def create_direct_boxplot(results_df, output_path, title, custom_labels):
    """
    Create boxplots with scatter points that directly follow the exact Excel order.
    This function creates two versions: one with log scale and one with linear scale.

    Parameters:
        results_df (DataFrame): DataFrame with the density results
        output_path (str): Path to save the output figure (without extension)
        title (str): Title for the plot
        custom_labels (list): Exact list of labels in the desired order from Excel
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Get unique image names
    image_names = results_df['image_name'].unique()

    # Natural sort the image names
    image_names = natural_sort(image_names)

    # Step 1: Create a map of image names to their corresponding label
    # Use custom_labels in their EXACT original order
    image_label_map = {}
    for i, image_name in enumerate(image_names):
        if i < len(custom_labels):
            image_label_map[image_name] = custom_labels[i]

    # Step 2: Organize data by image name for direct plotting
    # This avoids pandas' automatic ordering
    image_data = {}
    for image_name in image_names:
        # Get all density values for this image
        image_data[image_name] = results_df[results_df['image_name'] == image_name]['density'].values

    # Create manual boxplot in the EXACT order of custom_labels
    box_positions = []
    box_labels = []
    all_data = []

    # Track which images we've already plotted to handle duplicates
    processed_images = set()

    # Follow the exact order in custom_labels
    for i, label in enumerate(custom_labels):
        # Find the image that maps to this label
        for image_name, mapped_label in image_label_map.items():
            # If we find a match AND we haven't processed this image yet
            if mapped_label == label and image_name not in processed_images:
                # Add data for this image
                all_data.append(image_data[image_name])
                # Record the position
                box_positions.append(i + 1)  # 1-based positions
                # Add the label
                box_labels.append(label)
                # Mark this image as processed
                processed_images.add(image_name)
                # Stop looking for this label
                break

    # Define common properties for both plots
    boxprops = dict(linewidth=1.5, color='black')
    medianprops = dict(linewidth=2.0, color='#323232')

    # Generate colors for each box - similar to the Tab10 colormap
    import matplotlib.cm as cm
    colors = cm.tab10(np.linspace(0, 1, len(set(custom_labels))))

    # Map each box to a color based on its label
    unique_labels = list(set(custom_labels))
    color_map = {label: colors[i % len(colors)] for i, label in enumerate(unique_labels)}

    # Function to create a single plot with the specified scale
    def create_plot(scale_type, output_file):
        # Set up the figure
        plt.figure(figsize=(14, 8))

        # Extract background color from current style for consistent look
        bg_color = plt.rcParams.get('axes.facecolor', '#f0f0f0')
        plt.gca().set_facecolor(bg_color)

        # Create the boxplot with exact positioning
        bp = plt.boxplot(all_data, positions=box_positions, widths=0.6,
                        patch_artist=True, boxprops=boxprops, medianprops=medianprops)

        # Apply colors to boxes
        for i, patch in enumerate(bp['boxes']):
            label = box_labels[i]
            color_idx = unique_labels.index(label)
            patch.set_facecolor(colors[color_idx % len(colors)])
            patch.set_alpha(0.8)

        # Add scatter points for individual data points
        for i, data in enumerate(all_data):
            # Get position for this box
            pos = box_positions[i]
            # Add jitter to x position
            jitter = np.random.normal(0, 0.05, size=len(data))
            # Plot the data points
            plt.scatter(jitter + pos, data, color='black', alpha=0.6, s=40)

        # Set y-axis scale according to parameter
        if scale_type == 'log':
            plt.yscale('log')
            scale_title = title + " (Log Scale)"
        else:
            plt.yscale('linear')
            scale_title = title + " (Linear Scale)"

        # Set the x-ticks to the exact positions we used
        plt.xticks(box_positions, box_labels, rotation=45, ha='right')

        # Add grid, title, and labels
        plt.grid(True, axis='y', alpha=0.3)
        plt.title(scale_title, fontsize=16)
        plt.xlabel('Sample', fontsize=14)
        plt.ylabel('Predicted Density', fontsize=14)

        # Ensure the figure looks good
        plt.tight_layout()

        # Save the figure
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

    # Create both log and linear scale versions
    log_output_path = output_path.replace('.png', '_log.png')
    linear_output_path = output_path.replace('.png', '_linear.png')

    # Generate both plots
    create_plot('log', log_output_path)
    create_plot('linear', linear_output_path)

    return log_output_path, linear_output_path

# 5. ORIGINAL Function to create boxplot with scatter overlay - KEPT FOR COMPATIBILITY
def create_boxplot_with_scatter(results_df, output_path, title, combine_labels=False):
    """
    Create a boxplot with scatter points for each image, keeping labels separate
    and maintaining the exact order from Excel.

    Parameters:
        results_df (DataFrame): DataFrame with results
        output_path (str): Path to save the output figure
        title (str): Title for the plot
        combine_labels (bool): This parameter is ignored - we always keep labels separate
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import numpy as np

    # Create a copy to avoid modifying the original dataframe
    plot_df = results_df.copy()

    # Always keep labels separate, regardless of what was passed
    combine_labels = False

    # If we have custom_label and order_index columns, use them for ordering
    if 'custom_label' in plot_df.columns and 'order_index' in plot_df.columns:
        # Create a unique identifier for each image (using image_name)
        # This ensures each image gets its own box, even with identical labels
        plot_df['plot_label'] = plot_df.apply(
            lambda row: f"{row['custom_label']}|{row['image_name']}",
            axis=1
        )

        # Create a display label (what appears on the x-axis)
        plot_df['display_label'] = plot_df['custom_label']

        # Group by the plot label and get the order index for each group
        # This maintains the exact order from Excel
        label_order_map = {}

        # Get a representative order_index for each unique plot_label
        for label, group in plot_df.groupby('plot_label'):
            # Use the first order_index for this group
            label_order_map[label] = group['order_index'].iloc[0]

        # Get all unique plot labels
        all_plot_labels = plot_df['plot_label'].unique()

        # Sort the plot labels by their order_index (Excel order)
        sorted_labels = sorted(all_plot_labels, key=lambda x: label_order_map[x])

        # Create label mapping for display (remove the image_name part)
        label_map = {label: label.split('|')[0] for label in sorted_labels}

        # Use the sorted labels as categories
        unique_labels = sorted_labels
    else:
        # No Excel labels, use image_name and create a simple sequential order
        plot_df['plot_label'] = plot_df['image_name']
        plot_df['display_label'] = plot_df['image_name']

        # Get unique image names in the order they appear
        unique_labels = plot_df['plot_label'].unique()

        # Create a simple label mapping (identical in this case)
        label_map = {label: label for label in unique_labels}

    # Create categorical type to enforce the exact order we want
    plot_df['plot_label'] = pd.Categorical(
        plot_df['plot_label'],
        categories=unique_labels,
        ordered=True
    )

    # Create the figure with appropriate styling
    plt.figure(figsize=(14, 8))

    # Use a safer approach to styling - checking available styles first
    try:
        # Try to use a seaborn style if available
        import matplotlib.style as mplstyle
        available_styles = mplstyle.available

        # Check if any seaborn styles are available
        seaborn_styles = [style for style in available_styles if 'seaborn' in style]
        if seaborn_styles:
            # Use the first available seaborn style
            plt.style.use(seaborn_styles[0])
        else:
            # Fallback to a default style that should be available
            plt.style.use('default')
    except Exception as e:
        print(f"  Warning: Could not set plot style: {str(e)}")
        # Continue without setting a style

    # Create boxplot with custom colors
    medianprops = dict(color='#323232')  # Dark gray median line
    box = sns.boxplot(x='plot_label', y='density', data=plot_df,
                      patch_artist=True,
                      medianprops=medianprops)

    # Get unique groups for coloring
    if 'custom_label' in plot_df.columns:
        unique_groups = plot_df['custom_label'].unique()
    else:
        unique_groups = plot_df['image_name'].unique()

    # Generate colors for each unique group (similar to particleTrackCircle's approach)
    import matplotlib.cm as cm
    colors = cm.tab10(np.linspace(0, 1, len(unique_groups)))

    # Map each box to its color
    for i, patch in enumerate(box.artists):
        if i < len(unique_labels):
            label = unique_labels[i].split('_')[0]  # Get the base label without the index
            color_idx = np.where(unique_groups == label)[0][0] % len(colors)
            patch.set_facecolor(colors[color_idx])
            patch.set_alpha(0.8)

    # Add scatter points
    sns.stripplot(x='plot_label', y='density', data=plot_df,
                 size=8, color='black', alpha=0.6)

    # Set y-axis to log scale
    plt.yscale('log')

    # Replace x-tick labels if we have a label mapping
    if label_map:
        categories = plot_df['plot_label'].cat.categories
        labels = [label_map.get(cat, str(cat)) for cat in categories]
        box.set_xticklabels(labels)

    # Customize plot
    plt.title(title, fontsize=16)
    plt.xlabel('Sample', fontsize=14)
    plt.ylabel('Predicted Density (log scale)', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # Save the figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    return output_path

# 5. Function to create heatmap visualization of predicted densities
def create_density_heatmap(image_name, predictions, grid_positions, grid_size, output_path):
    """
    Create a heatmap visualization showing the predicted densities for each grid cell
    """
    # Initialize empty grid with NaN values (for cells without predictions)
    heatmap_data = np.full((grid_size, grid_size), np.nan)

    # Fill in the grid with predictions
    for (row, col), density in zip(grid_positions, predictions):
        heatmap_data[row, col] = density

    # Create the heatmap
    plt.figure(figsize=(10, 8))
    masked_data = np.ma.masked_invalid(heatmap_data)  # Mask NaN values

    # Get magnification from image name for simplified title
    mag = "Unknown"
    if 'X_' in image_name or 'X ' in image_name:
        try:
            mag = image_name.split('X')[0].split('_')[-1].strip() + 'X'
        except:
            mag = image_name
    else:
        mag = image_name

    # Create heatmap
    ax = sns.heatmap(masked_data, annot=True, fmt=".1f", cmap="YlGnBu",
                    cbar_kws={'label': 'Predicted Density'})

    # Mark corners as excluded
    corners = [(0, 0), (0, grid_size-1), (grid_size-1, 0), (grid_size-1, grid_size-1)]
    for row, col in corners:
        ax.add_patch(plt.Rectangle((col, row), 1, 1, fill=False, edgecolor='red',
                                  lw=2, hatch='///'))

    # Use a simplified title to avoid character encoding issues
    plt.title(f"Density Heatmap for {mag}", fontsize=16)
    plt.tight_layout()

    # Save the figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    return output_path

# 6. Function to extract labels from CSV or Excel files in a subfolder - IMPROVED VERSION
def extract_labels_from_file(subfolder_path):
    """
    Look for Excel or CSV files in a subfolder and extract labels.
    Improved to avoid temporary Excel files and handle encoding issues.
    If not found, return None and a warning will be shown.
    """
    # Check for Excel files first, being careful to exclude temporary Excel files (start with ~$)
    excel_files = []
    for ext in ["*.xlsx", "*.xls"]:
        for file_path in Path(subfolder_path).glob(ext):
            # Skip temporary Excel files that start with ~$
            if not file_path.name.startswith("~$"):
                excel_files.append(file_path)

    # If Excel files found, try to read them
    if excel_files:
        try:
            # Try importing openpyxl
            try:
                import openpyxl
            except ImportError:
                print("  Warning: Missing dependency 'openpyxl'. Install with: pip install openpyxl")
                return None

            selected_file = excel_files[0]
            print(f"  Reading Excel file: {selected_file}")

            # Try different Excel engines to handle potential file format issues
            try:
                # First try with default engine
                df = pd.read_excel(selected_file, header=None)
            except Exception as e1:
                print(f"  First Excel reading attempt failed: {str(e1)}")
                try:
                    # Try with openpyxl engine explicitly
                    df = pd.read_excel(selected_file, header=None, engine='openpyxl')
                except Exception as e2:
                    print(f"  Second Excel reading attempt failed: {str(e2)}")
                    try:
                        # Try with xlrd engine for older Excel formats
                        df = pd.read_excel(selected_file, header=None, engine='xlrd')
                    except Exception as e3:
                        print(f"  All Excel reading attempts failed. Last error: {str(e3)}")
                        return None

            # Extract labels from the first column - PRESERVE EXACT ORDER
            if df.shape[0] > 0:
                labels = df.iloc[:, 0].tolist()
                labels = [str(label).strip() for label in labels if pd.notna(label)]

                # Debug info
                print(f"  Successfully extracted {len(labels)} labels from Excel in original order: {labels}")

                if labels:
                    return labels
            else:
                print(f"  Excel file is empty or has no data in the first column")

        except Exception as e:
            print(f"  Error reading Excel file: {str(e)}")
            import traceback
            traceback.print_exc()
    else:
        print(f"  No valid Excel files found in {subfolder_path}")

        # If no Excel files, try looking for a CSV file
        csv_files = list(Path(subfolder_path).glob("*.csv"))
        if csv_files:
            try:
                print(f"  Trying CSV file instead: {csv_files[0]}")
                # Try different encodings for CSV
                encodings = ['utf-8', 'latin1', 'cp1252', 'gb18030', 'big5']

                for encoding in encodings:
                    try:
                        df = pd.read_csv(csv_files[0], header=None, encoding=encoding)

                        # Extract labels from the first column - PRESERVE EXACT ORDER
                        labels = df.iloc[:, 0].tolist()
                        labels = [str(label).strip() for label in labels if pd.notna(label)]

                        if labels:
                            print(f"  Successfully read CSV with encoding {encoding}")
                            print(f"  Labels from CSV in original order: {labels}")
                            return labels
                    except Exception as csv_error:
                        continue

                print("  Could not read CSV file with any encoding")
            except Exception as e:
                print(f"  Error reading CSV file: {str(e)}")

    # If we get here, we couldn't find or read any Excel or CSV files
    print("  No valid Excel or CSV files found with labels. Falling back to default behavior.")
    return None
# Main function
def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Predict microbead densities on new images')
    parser.add_argument('--model_path', type=str, default="model/best_model_batch32_filters128-256-512.pth",
                        help='Path to the trained model file')
    parser.add_argument('--input_dir', type=str, default="new_images",
                        help='Directory containing new images to predict')
    parser.add_argument('--output_dir', type=str, default="prediction_results",
                        help='Directory to save prediction results')
    parser.add_argument('--crops_dir', type=str, default="prediction_crops",
                        help='Directory to save cropped images')
    parser.add_argument('--grid_size', type=int, default=4,
                        help='Size of the grid (e.g., 4 for a 4x4 grid)')
    parser.add_argument('--crop_size', type=int, default=512,
                        help='Size of each crop in pixels')
    parser.add_argument('--create_heatmaps', action='store_true',
                        help='Create heatmap visualizations for each image (default: False)')
    parser.add_argument('--save_crops', action='store_true',
                        help='Save cropped images (default: False)')
    parser.add_argument('--combine_labels', action='store_true',
                        help='Combine data points with identical labels (default: False)')

    args = parser.parse_args()

    # Create timestamp for unique output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(args.output_dir, f"pred_{timestamp}")
    crops_dir = os.path.join(args.crops_dir, f"crops_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)

    # Only create crops directory if save_crops is enabled
    if args.save_crops:
        os.makedirs(crops_dir, exist_ok=True)

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else
                          "cpu")
    print(f"Using device: {device}")

    # Load the trained model
    filter_config = [128, 256, 512]  # Must match the trained model configuration
    model = DensityRegressionCNN(filters=filter_config)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"Loaded model from {args.model_path}")

    # Setup image preprocessing transform (same as used for evaluation)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    # Note about combine_labels parameter - we're keeping this for backwards compatibility
    # but the modified boxplot function will always keep labels separate
    print(f"Using separate labels for each image, even if they have identical text")

    # Get all subfolders in the input directory
    input_path = Path(args.input_dir)
    if input_path.is_dir():
        subfolders = [d for d in input_path.iterdir() if d.is_dir()]
        # If no subfolders, treat the input_dir itself as the only "subfolder"
        if not subfolders:
            subfolders = [input_path]
    else:
        print(f"Error: Input directory {args.input_dir} does not exist.")
        return

    print(f"Found {len(subfolders)} subfolders/collections to process")

    # Initialize a dictionary to store results by subfolder
    all_results_by_subfolder = {}

    # Process each subfolder
    for subfolder in subfolders:
        subfolder_name = subfolder.name
        print(f"\nProcessing subfolder: {subfolder_name}")

        # Create subdirectories for this subfolder's results and crops
        subfolder_results_dir = os.path.join(results_dir, subfolder_name)
        subfolder_crops_dir = os.path.join(crops_dir, subfolder_name)
        os.makedirs(subfolder_results_dir, exist_ok=True)
        os.makedirs(subfolder_crops_dir, exist_ok=True)

        # Find all images in this subfolder - use os.walk to handle Chinese characters better
        image_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff']
        image_paths = []

        # Use os.walk which handles non-ASCII characters better than Path.glob in some environments
        for root, dirs, files in os.walk(str(subfolder)):
            for file in files:
                file_lower = file.lower()
                if any(file_lower.endswith(ext) for ext in image_extensions):
                    image_paths.append(os.path.join(root, file))

        # Convert to Path objects
        image_paths = [Path(p) for p in image_paths]

        print(f"  Found {len(image_paths)} images to process in {subfolder_name}")

        if not image_paths:
            print(f"  No images found in {subfolder_name}, skipping.")
            continue

        # Sort the image paths with natural sorting
        image_paths = natural_sort(image_paths, key_func=lambda x: x.stem)

        print(f"  Sorted image order (natural): {[path.stem for path in image_paths]}")

        # Initialize results list for this subfolder
        subfolder_results = []

        # Process each image in the subfolder in sorted order
        for img_path in image_paths:
            image_name = img_path.stem
            print(f"  Processing image: {image_name}")

            # Create subdirectory for this image's crops if save_crops is enabled
            image_crops_dir = None
            if args.save_crops:
                image_crops_dir = os.path.join(subfolder_crops_dir, image_name)
                os.makedirs(image_crops_dir, exist_ok=True)
            else:
                # Create a temporary directory for crops if not saving them
                image_crops_dir = os.path.join(tempfile.gettempdir(), f"temp_crops_{timestamp}_{image_name}")
                os.makedirs(image_crops_dir, exist_ok=True)

            # Crop the image into grid cells
            crop_paths, crop_positions, viz_path = crop_center_grid(
                str(img_path), image_crops_dir, args.grid_size, args.crop_size, args.save_crops
            )

            print(f"    Created {len(crop_paths)} crops")

            # Predict density for each crop
            predictions = []
            valid_positions = []
            for crop_path, position in zip(crop_paths, crop_positions):
                density = predict_density(model, crop_path, transform, device)
                if density is not None:
                    predictions.append(density)
                    valid_positions.append(position)
                    print(f"    Crop at position {position}: Predicted density = {density:.2f}")

                    # Add to results
                    subfolder_results.append({
                        'subfolder': subfolder_name,
                        'image_name': image_name,
                        'crop_name': os.path.basename(crop_path),
                        'row': position[0],
                        'col': position[1],
                        'density': density
                    })

            # Create heatmap visualization if enabled
            if args.create_heatmaps and predictions:
                heatmap_path = os.path.join(subfolder_results_dir, f"{image_name}_density_heatmap.png")
                create_density_heatmap(image_name, predictions, valid_positions,
                                     args.grid_size, heatmap_path)
                print(f"    Created heatmap at {heatmap_path}")

            # Clean up temporary crop directory if not saving crops
            if not args.save_crops:
                try:
                    import shutil
                    shutil.rmtree(image_crops_dir)
                except:
                    pass

        # Convert results to dataframe for this subfolder
        if subfolder_results:
            subfolder_df = pd.DataFrame(subfolder_results)
            all_results_by_subfolder[subfolder_name] = subfolder_df

            # Calculate summary statistics for this subfolder
            summary_df = subfolder_df.groupby('image_name').agg(
                mean_density=('density', 'mean'),
                median_density=('density', 'median'),
                std_density=('density', 'std'),
                min_density=('density', 'min'),
                max_density=('density', 'max'),
                crop_count=('density', 'count')
            ).reset_index()

            # Save results to CSV for this subfolder
            results_path = os.path.join(subfolder_results_dir, f"{subfolder_name}_predictions.csv")
            summary_path = os.path.join(subfolder_results_dir, f"{subfolder_name}_summary.csv")
            subfolder_df.to_csv(results_path, index=False)
            summary_df.to_csv(summary_path, index=False)

            print(f"  Saved predictions to {results_path}")
            print(f"  Saved summary to {summary_path}")

            # Try to extract labels from any CSV/Excel files in the subfolder
            custom_labels = extract_labels_from_file(subfolder)

            # Create boxplot with scatter points for this subfolder
            boxplot_path = os.path.join(subfolder_results_dir, f"{subfolder_name}_density_boxplot.png")

            # Debug information about image names and labels
            image_names = natural_sort(subfolder_df['image_name'].unique())
            print(f"  Image names found (natural sorted): {image_names}")
            print(f"  Custom labels found: {custom_labels}")

            # If we have valid custom labels from Excel, use the direct boxplot function
            if custom_labels and len(custom_labels) >= len(image_names):
                print(f"  Creating boxplot with EXACT Excel order: {custom_labels}")

                # Use the direct boxplot function that follows Excel order exactly
                boxplot_log_path, boxplot_linear_path = create_direct_boxplot(
                    subfolder_df,
                    boxplot_path,
                    f"Microbead Density Distribution - {subfolder_name}",
                    custom_labels
                )

                print(f"  Created log scale boxplot at: {boxplot_log_path}")
                print(f"  Created linear scale boxplot at: {boxplot_linear_path}")
            else:
                # Fall back to the standard boxplot if we don't have proper labels
                print(f"  No valid Excel labels found, using standard boxplot")

                # Prepare data for the standard boxplot
                boxplot_df = subfolder_df.copy()

                # Create boxplot with the standard function (log scale only)
                boxplot_path = create_boxplot_with_scatter(
                    boxplot_df,
                    boxplot_path,
                    f"Microbead Density Distribution - {subfolder_name}",
                    False  # Always keep labels separate
                )

                print(f"  Created boxplot visualization at {boxplot_path}")

            print(f"  Created boxplot visualization at {boxplot_path}")

    # Combine all results into a single dataframe
    all_results = pd.concat(all_results_by_subfolder.values()) if all_results_by_subfolder else pd.DataFrame()

    if not all_results.empty:
        # Save combined results
        combined_results_path = os.path.join(results_dir, "all_predictions.csv")
        all_results.to_csv(combined_results_path, index=False)

        # Calculate summary statistics across all images and subfolders
        combined_summary_df = all_results.groupby(['subfolder', 'image_name']).agg(
            mean_density=('density', 'mean'),
            median_density=('density', 'median'),
            std_density=('density', 'std'),
            min_density=('density', 'min'),
            max_density=('density', 'max'),
            crop_count=('density', 'count')
        ).reset_index()

        combined_summary_path = os.path.join(results_dir, "summary_statistics.csv")
        combined_summary_df.to_csv(combined_summary_path, index=False)

        print(f"\nSaved all predictions to {combined_results_path}")
        print(f"Saved summary statistics to {combined_summary_path}")

        # Create a combined boxplot
        combined_boxplot_path = os.path.join(results_dir, "combined_density_boxplot.png")

        # Collect all custom labels across all subfolders to maintain global order
        all_custom_labels = []
        for subfolder in subfolders:
            subfolder_labels = extract_labels_from_file(subfolder)
            if subfolder_labels:
                all_custom_labels.extend(subfolder_labels)

        # If we have custom labels, use the direct boxplot method
        if all_custom_labels:
            print(f"Creating combined boxplot with all labels in exact order")
            combined_log_path, combined_linear_path = create_direct_boxplot(
                all_results,
                combined_boxplot_path,
                "Microbead Density Distribution - All Samples",
                all_custom_labels
            )
            print(f"Created combined log scale visualization at {combined_log_path}")
            print(f"Created combined linear scale visualization at {combined_linear_path}")
        else:
            # Fall back to standard method
            combined_boxplot_path = create_boxplot_with_scatter(
                all_results,
                combined_boxplot_path,
                "Microbead Density Distribution - All Samples",
                False  # Always use False to keep labels separate
            )
            print(f"Created combined visualization at {combined_boxplot_path}")

        print(f"Created combined visualization at {combined_boxplot_path}")

        # Calculate average density across all images
        avg_density = all_results['density'].mean()
        print(f"\nAverage predicted density across all images: {avg_density:.2f}")
    else:
        print("\nNo results were generated. Check your input directory and image files.")

    # Save configuration
    config = {
        'model_path': args.model_path,
        'filter_config': filter_config,
        'grid_size': args.grid_size,
        'crop_size': args.crop_size,
        'input_dir': args.input_dir,
        'subfolder_count': len(subfolders),
        'create_heatmaps': args.create_heatmaps,
        'save_crops': args.save_crops,
        'combine_labels': args.combine_labels,
        'timestamp': timestamp,
        'average_density': float(avg_density) if 'avg_density' in locals() else None
    }

    with open(os.path.join(results_dir, "prediction_config.json"), 'w') as f:
        json.dump(config, f, indent=4)

    print(f"\nPrediction complete. All results saved to {results_dir}")

if __name__ == "__main__":
    main()
