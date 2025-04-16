
#!/usr/bin/env python
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

# 2. Function to crop images in a grid pattern (copied from cropImg_center.py)
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
        return []

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
        cv2.imwrite(viz_path, viz_img)
        print(f"Created visualization image: {viz_path}")

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

# 4. Function to create boxplot with scatter overlay
def create_boxplot_with_scatter(results_df, output_path, title):
    """
    Create a boxplot combined with scatter points for each image
    """
    # Create a copy to avoid modifying the original dataframe
    plot_df = results_df.copy()

    # Extract magnification information from image names (e.g., 5X, 10X, etc.)
    def extract_magnification(name):
        if 'X_' in name or 'X ' in name:
            parts = name.split('X')[0]
            # Try to extract the number before X
            try:
                mag = parts.split('_')[-1].strip()
                return f"{mag}X"
            except:
                return name
        return name

    # Apply the magnification extraction
    plot_df['display_name'] = plot_df['image_name'].apply(extract_magnification)

    # Sort naturally by the display name (magnification)
    import re
    def natural_sort_key(s):
        return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

    # Get unique display names, naturally sorted
    unique_display_names = sorted(plot_df['display_name'].unique(), key=natural_sort_key)

    # Create a categorical type with the sorted order
    plot_df['display_name'] = pd.Categorical(
        plot_df['display_name'],
        categories=unique_display_names,
        ordered=True
    )

    plt.figure(figsize=(12, 8))

    # Create boxplot with simplified names
    box = sns.boxplot(x='display_name', y='density', data=plot_df, palette='Set3')

    # Add scatter points
    sns.stripplot(x='display_name', y='density', data=plot_df,
                 size=8, color='black', alpha=0.6)

    # Set y-axis to log scale
    plt.yscale('log')

    # Customize plot
    plt.title(title, fontsize=16)
    plt.xlabel('Magnification', fontsize=14)
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

    args = parser.parse_args()

    # Create timestamp for unique output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(args.output_dir, f"pred_{timestamp}")
    crops_dir = os.path.join(args.crops_dir, f"crops_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
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

    # Find all images in input directory
    image_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(list(Path(args.input_dir).glob(f"*{ext}")))

    print(f"Found {len(image_paths)} images to process")

    # Process each image
    all_results = []
    for img_path in image_paths:
        image_name = img_path.stem
        print(f"\nProcessing image: {image_name}")

        # Create subdirectory for this image's crops
        image_crops_dir = os.path.join(crops_dir, image_name)
        os.makedirs(image_crops_dir, exist_ok=True)

        # Crop the image into grid cells
        crop_paths, crop_positions, viz_path = crop_center_grid(
            str(img_path), image_crops_dir, args.grid_size, args.crop_size, True
        )

        print(f"Created {len(crop_paths)} crops")

        # Predict density for each crop
        predictions = []
        valid_positions = []
        for crop_path, position in zip(crop_paths, crop_positions):
            density = predict_density(model, crop_path, transform, device)
            if density is not None:
                predictions.append(density)
                valid_positions.append(position)
                print(f"  Crop at position {position}: Predicted density = {density:.2f}")

                # Add to results
                all_results.append({
                    'image_name': image_name,
                    'crop_name': os.path.basename(crop_path),
                    'row': position[0],
                    'col': position[1],
                    'density': density
                })

        # Create heatmap visualization
        heatmap_path = os.path.join(results_dir, f"{image_name}_density_heatmap.png")
        create_density_heatmap(image_name, predictions, valid_positions,
                              args.grid_size, heatmap_path)

    # Convert results to dataframe
    results_df = pd.DataFrame(all_results)

    # Calculate summary statistics for each image
    summary_df = results_df.groupby('image_name').agg(
        mean_density=('density', 'mean'),
        median_density=('density', 'median'),
        std_density=('density', 'std'),
        min_density=('density', 'min'),
        max_density=('density', 'max'),
        crop_count=('density', 'count')
    ).reset_index()

    # Save results to CSV
    results_path = os.path.join(results_dir, "all_predictions.csv")
    summary_path = os.path.join(results_dir, "summary_statistics.csv")
    results_df.to_csv(results_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print(f"\nSaved all predictions to {results_path}")
    print(f"Saved summary statistics to {summary_path}")

    # Create boxplot with scatter points
    boxplot_path = os.path.join(results_dir, "density_boxplot.png")
    create_boxplot_with_scatter(results_df, boxplot_path,
                               "Microbead Density Distribution by Image")

    print(f"Created visualization at {boxplot_path}")

    # Calculate average density across all images
    avg_density = results_df['density'].mean()
    print(f"\nAverage predicted density across all images: {avg_density:.2f}")

    # Save configuration
    config = {
        'model_path': args.model_path,
        'filter_config': filter_config,
        'grid_size': args.grid_size,
        'crop_size': args.crop_size,
        'image_count': len(image_paths),
        'timestamp': timestamp,
        'average_density': float(avg_density)
    }

    with open(os.path.join(results_dir, "prediction_config.json"), 'w') as f:
        json.dump(config, f, indent=4)

    print(f"\nPrediction complete. All results saved to {results_dir}")

if __name__ == "__main__":
    main()
