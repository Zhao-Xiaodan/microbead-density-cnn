
# Microbead Density CNN

A deep learning framework for automated analysis and prediction of microbead density from microscopy images.

## Overview

This project implements a Convolutional Neural Network (CNN) to analyze microscopy images of microbeads and predict their density. The workflow includes image cropping, data preparation, model training, evaluation, and prediction on new images.

## Table of Contents

- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Prediction](#prediction)
- [File Descriptions](#file-descriptions)
- [Usage Examples](#usage-examples)

## Installation

### MacOS (M1/M2)

Create and activate the conda environment using the provided environment file:

```bash
conda env create -f environment_cnn.yml
conda activate microbead-cnn
```

### HPC or Other Systems

Create and activate the conda environment manually:

```bash
conda create -n microbead-cnn python=3.9
conda init bash
source ~/.bashrc
conda activate microbead-cnn  # On some systems: source activate microbead-cnn

# Install required packages
conda install -y numpy pandas matplotlib pillow scikit-learn opencv jpeg seaborn
pip install torch torchvision tqdm pathlib
```

## Dataset Preparation

The dataset preparation pipeline involves two main steps: image cropping and density mapping.

### 1. Image Cropping

The script `cropImg_grid.py` divides original microscopy images into a grid of crops, excluding the four corner cells:

```bash
python cropImg_grid.py --input original_images --output cropped_images --grid_size 4 --crop_size 512 --visualize
```

This will:
- Load images from the `original_images` directory
- Create a 4×4 grid for each image
- Generate 512×512 pixel crops from each grid cell (excluding corners)
- Save crops to the `cropped_images` directory
- Generate visualization images showing the grid layout

### 2. Density Mapping

After cropping, use `match_cropped_images.py` to map each cropped image to its corresponding ground truth density:

```bash
python match_cropped_images.py
```

This script:
- Reads the original density values from `beadDensity_with_filenames.csv`
- Matches each cropped image to its parent image and associated density
- Generates `cropped_images_with_density.csv` containing filename-density pairs

### 3. Dataset Organization

Organize your dataset folder structure as follows:

```
dataset/
├── images/
│   ├── image1_grid_r0_c1.png
│   ├── image1_grid_r0_c2.png
│   └── ...
└── density.csv
```

The `density.csv` file should contain two columns:
- Column 1: Cropped image filename
- Column 2: Ground truth density value

## Training

### Local Training

For training on a local machine:

```bash
python train_microbead_cnn_hpc.py --batch_sizes 16 32 64 --filter_configs "32,64,128" "64,128,256" "128,256,512" --epochs 40 --patience 10 --learning_rate 0.001 --output_dir training_results
```

### HPC Training

For training on an HPC cluster:

1. Modify the `pbs_microbead-cnn.sh` script if needed to adjust resource requirements
2. Submit the job:

```bash
qsub pbs_microbead-cnn.sh
```

The script will train multiple CNN configurations with different batch sizes and filter combinations, and save the results in the specified output directory.

### Hyperparameter Optimization

The training script experiments with:
- Different batch sizes (16, 32, 64)
- Various filter configurations ("32,64,128", "16,32,64", "64,128,256", "128,256,512")
- Early stopping with a patience parameter
- Learning rate adjustments

## Evaluation

After training, analyze and compare the results of different model configurations using:

```bash
python plot_batch_filter.py
```

This script:
- Loads the experiment comparison data from the training results
- Creates visualizations comparing model performance across different metrics (MSE, MAE, R²)
- Identifies the best model configuration based on accuracy and training time
- Generates plots showing the relationships between hyperparameters and performance

## Prediction

Use the best trained model to predict microbead density on new images:

```bash
python predict_microbead_grid.py --model_path model/best_model_batch32_filters128-256-512.pth --input_dir new_images --output_dir prediction_results --crops_dir prediction_crops --grid_size 4 --crop_size 512 --create_heatmaps --save_crops
```

The prediction script:
- Loads the specified trained model
- Processes new images by dividing them into grid cells
- Predicts density for each valid crop
- Generates summary statistics and visualizations
- Creates density heatmaps for spatial distribution analysis
- Produces boxplots comparing density distributions across samples

## File Descriptions

### Core Scripts

- **cropImg_grid.py**: Crops original images into grid cells, excluding corners
- **match_cropped_images.py**: Maps cropped images to their density values
- **train_microbead_cnn_hpc.py**: Trains multiple CNN configurations
- **plot_batch_filter.py**: Analyzes training results to identify optimal configurations
- **predict_microbead_grid.py**: Applies trained model to new images

### Support Files

- **pbs_microbead-cnn.sh**: PBS job submission script for HPC training
- **environment_cnn.yml**: Conda environment specification for MacOS M1/M2

## Usage Examples

### Complete Workflow

```bash
# 1. Prepare the environment
conda activate microbead-cnn

# 2. Crop input images
python cropImg_grid.py --input original_images --output cropped_images

# 3. Match crops with density values
python match_cropped_images.py

# 4. Train models (HPC)
qsub pbs_microbead-cnn.sh

# 5. Analyze results and select best model
python plot_batch_filter.py

# 6. Predict on new images
python predict_microbead_grid.py --model_path model/best_model.pth --input_dir new_images
```

## Model Architecture

The CNN architecture for density regression consists of:
- Three convolutional blocks with batch normalization and ReLU activation
- Max pooling layers after each convolutional block
- Global average pooling to reduce spatial dimensions
- Two fully connected layers with dropout regularization
- MSE loss function for regression training

## Results Interpretation

The prediction results include:
- CSV files with predicted densities for each crop
- Summary statistics for each image and collection
- Boxplots showing density distributions
- Heatmaps visualizing spatial density variations

The boxplots can be interpreted in two ways:
1. **Log scale**: Better for visualizing wide density ranges
2. **Linear scale**: Better for direct comparison between samples

## Advanced Features

- **Custom Labels**: The prediction script can read labels from Excel or CSV files in the input folder
- **Batch Processing**: Multiple image collections can be processed in a single run
- **Visualization Options**: Generate heatmaps and boxplots with various styling options
- **Early Stopping**: Training with patience to prevent overfitting

## License

This project is provided for research and educational purposes.

## Acknowledgments

This work implements deep learning approaches for automated analysis of microbead density in microscopy images.
