
#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from PIL import Image
import os
import pandas as pd
import matplotlib.pyplot as plt
import copy
import time
import json
import argparse
from datetime import datetime

# Create argument parser for configuration options
parser = argparse.ArgumentParser(description='Train a CNN for microbead density regression')
parser.add_argument('--input_dir', type=str, default='dataset',
                    help='Input directory containing images/ and density.csv (default: dataset)')
parser.add_argument('--batch_sizes', nargs='+', type=int, default=[16, 32, 64],
                    help='Batch sizes to experiment with')
parser.add_argument('--filter_configs', nargs='+', type=str,
                    default=['32,64,128', '16,32,64', '64,128,256'],
                    help='Comma-separated filter configurations for each layer')
parser.add_argument('--epochs', type=int, default=40,
                    help='Maximum number of epochs to train')
parser.add_argument('--patience', type=int, default=10,
                    help='Early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.001,
                    help='Learning rate for optimizer')
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed for reproducibility')
parser.add_argument('--output_dir', type=str, default='training_results',
                    help='Directory to save results')
args = parser.parse_args()

# Set random seeds for reproducibility
torch.manual_seed(args.seed)
np.random.seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

# 1. Define a configurable CNN Model
class DensityRegressionCNN(nn.Module):
    def __init__(self, filters=[32, 64, 128]):
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

# 2. Custom Dataset Class for CSV-based density data
class MicrobeadDataset(Dataset):
    def __init__(self, image_dir, density_csv, transform=None):
        """
        image_dir: Directory with image files (e.g., '/dataset/images/')
        density_csv: CSV file with image filenames and density values
        transform: Image preprocessing steps
        """
        self.image_dir = image_dir
        self.transform = transform

        # Load density data from CSV
        self.df = pd.read_csv(density_csv)

        # Check if CSV has proper columns
        if len(self.df.columns) == 1:
            # If CSV has a single column with space-separated values
            self.df = self.df.iloc[:, 0].str.split(expand=True)
            self.df.columns = ['filename', 'density']
        elif len(self.df.columns) == 2:
            # If CSV has separate columns
            self.df.columns = ['filename', 'density']
        else:
            raise ValueError("CSV format not recognized. Should have filename and density columns.")

        # Convert density to float
        self.df['density'] = self.df['density'].astype(float)

        print(f"Loaded {len(self.df)} image-density pairs from CSV")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx]['filename']
        # Add extension if not present
        if not img_name.endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff')):
            img_name = img_name + '.png'  # or whatever extension your images use
        density = self.df.iloc[idx]['density']

        img_path = os.path.join(self.image_dir, img_name)
        try:
            image = Image.open(img_path).convert('L')  # Grayscale
            if self.transform:
                image = self.transform(image)
            return image, torch.tensor(density, dtype=torch.float32)
        except Exception as e:
            print(f"Error loading image {img_path}: {str(e)}")
            # Return a dummy image and the density in case of error
            dummy_image = torch.zeros((1, 224, 224), dtype=torch.float32)
            return dummy_image, torch.tensor(density, dtype=torch.float32)

# 3. Data Preprocessing and Augmentation
# Training transform with augmentation
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
    # Data augmentation
    transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
])

# Transform for evaluation (no augmentation)
eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

# 4. Training Function with early stopping and best model saving
def train_model(model, train_loader, val_loader, config, device='cuda'):
    """
    Train the model with the given configuration
    """
    # Unpack configuration
    batch_size = config['batch_size']
    filter_config = config['filter_config']
    num_epochs = config['num_epochs']
    patience = config['patience']
    learning_rate = config['learning_rate']

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.to(device)

    # For tracking metrics
    history = {
        'train_loss': [],
        'val_loss': []
    }

    # For early stopping
    best_val_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    no_improve_epochs = 0
    epochs_completed = 0

    # Add memory tracking
    if device == torch.device('cuda'):
        peak_memory = 0

    start_time = time.time()

    for epoch in range(num_epochs):
        epochs_completed = epoch + 1

        # Training phase
        model.train()
        train_loss = 0.0
        for images, densities in train_loader:
            images, densities = images.to(device), densities.to(device)
            optimizer.zero_grad()
            outputs = model(images).squeeze()
            loss = criterion(outputs, densities)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)

            # Track GPU memory usage
            if device == torch.device('cuda'):
                memory_used = torch.cuda.max_memory_allocated() / (1024 ** 3)  # Convert to GB
                peak_memory = max(peak_memory, memory_used)

        train_loss /= len(train_loader.dataset)
        history['train_loss'].append(train_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, densities in val_loader:
                images, densities = images.to(device), densities.to(device)
                outputs = model(images).squeeze()
                loss = criterion(outputs, densities)
                val_loss += loss.item() * images.size(0)
            val_loss /= len(val_loader.dataset)
            history['val_loss'].append(val_loss)

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        # Check if this is the best model so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            no_improve_epochs = 0
            # Save the current best model
            model_save_path = os.path.join(
                config['output_dir'],
                f"best_model_batch{batch_size}_filters{'-'.join(map(str, filter_config))}.pth"
            )
            torch.save(model.state_dict(), model_save_path)
            print(f"New best model saved with validation loss: {val_loss:.4f}")
        else:
            no_improve_epochs += 1
            print(f"No improvement for {no_improve_epochs} epochs")

        # Early stopping check
        if no_improve_epochs >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

    # Calculate training time
    time_elapsed = time.time() - start_time
    training_minutes = time_elapsed / 60

    # Memory usage stats
    if device == torch.device('cuda'):
        memory_stats = {
            'peak_memory_gb': peak_memory
        }
    else:
        memory_stats = {'peak_memory_gb': None}

    print(f'Training completed in {training_minutes:.2f} minutes')
    print(f'Best validation loss: {best_val_loss:.4f}')

    # Load the best model weights
    model.load_state_dict(best_model_wts)

    # Plot training curves
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.title(f'Training and Validation Loss\nBatch: {batch_size}, Filters: {filter_config}')
    plt.legend()
    plot_save_path = os.path.join(
        config['output_dir'],
        f"training_curve_batch{batch_size}_filters{'-'.join(map(str, filter_config))}.png"
    )
    plt.savefig(plot_save_path)
    plt.close()

    # Save training history
    training_stats = {
        'config': {
            'batch_size': batch_size,
            'filter_config': filter_config,
            'learning_rate': learning_rate
        },
        'performance': {
            'epochs_completed': epochs_completed,
            'best_val_loss': float(best_val_loss),  # Convert to float for JSON serialization
            'training_minutes': training_minutes,
            **memory_stats
        },
        'history': {
            'train_loss': [float(x) for x in history['train_loss']],  # Convert to float for JSON serialization
            'val_loss': [float(x) for x in history['val_loss']]
        }
    }

    return model, training_stats, epochs_completed

# Function to evaluate model performance
def evaluate_model(model, test_loader, config, device):
    """
    Evaluate the model and return performance metrics
    """
    batch_size = config['batch_size']
    filter_config = config['filter_config']

    model.eval()
    predictions = []
    actual_values = []
    filenames = []

    with torch.no_grad():
        for batch_idx, (images, densities) in enumerate(test_loader):
            images, densities = images.to(device), densities.to(device)
            outputs = model(images).squeeze()

            predictions.extend(outputs.cpu().numpy())
            actual_values.extend(densities.cpu().numpy())

            # Get filenames for the current batch - FIXED VERSION
            # We can't directly access indices, so we'll just use batch position
            for i in range(len(images)):
                # Create a simple filename identifier based on batch position
                filename = f"test_sample_{batch_idx*batch_size + i}"
                filenames.append(filename)

    # Rest of the function remains the same...
    # Convert to numpy arrays
    predictions = np.array(predictions)
    actual_values = np.array(actual_values)

    # Calculate metrics
    mse = np.mean((predictions - actual_values) ** 2)
    mae = np.mean(np.abs(predictions - actual_values))
    r2 = 1 - (np.sum((actual_values - predictions) ** 2) /
              np.sum((actual_values - np.mean(actual_values)) ** 2))

    print(f"Model Evaluation Metrics (Batch: {batch_size}, Filters: {filter_config}):")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"R² Score: {r2:.4f}")

    # Plot predictions vs actual
    plt.figure(figsize=(10, 10))
    plt.scatter(actual_values, predictions, alpha=0.7)
    plt.plot([min(actual_values), max(actual_values)],
             [min(actual_values), max(actual_values)], 'r--')
    plt.xlabel('Actual Density')
    plt.ylabel('Predicted Density')
    plt.title(f'Predicted vs Actual Density\nBatch: {batch_size}, Filters: {filter_config}')
    plot_save_path = os.path.join(
        config['output_dir'],
        f"prediction_performance_batch{batch_size}_filters{'-'.join(map(str, filter_config))}.png"
    )
    plt.savefig(plot_save_path)
    plt.close()

    # Create results CSV with filename, actual, and predicted values
    results_df = pd.DataFrame({
        'filename': filenames,
        'actual_density': actual_values,
        'predicted_density': predictions,
        'absolute_error': np.abs(actual_values - predictions)
    })
    # Sort by error to identify problematic predictions
    results_df = results_df.sort_values('absolute_error', ascending=False)
    csv_save_path = os.path.join(
        config['output_dir'],
        f"test_predictions_batch{batch_size}_filters{'-'.join(map(str, filter_config))}.csv"
    )
    results_df.to_csv(csv_save_path, index=False)

    # Return evaluation metrics
    eval_metrics = {
        'mse': float(mse),
        'mae': float(mae),
        'r2': float(r2)
    }

    return eval_metrics

# Main Execution
if __name__ == "__main__":
    # Set paths using the configurable input directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(base_dir, args.input_dir)
    image_dir = os.path.join(input_path, 'images')
    density_file = os.path.join(input_path, 'density.csv')

    # Validate that the input directory and files exist
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input directory does not exist: {input_path}")
    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"Images directory does not exist: {image_dir}")
    if not os.path.exists(density_file):
        raise FileNotFoundError(f"Density CSV file does not exist: {density_file}")

    print(f"Using input directory: {input_path}")
    print(f"Images directory: {image_dir}")
    print(f"Density file: {density_file}")

    # Create timestamp for unique output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # Save configuration parameters
    config_summary = {
        'input_dir': args.input_dir,
        'batch_sizes': args.batch_sizes,
        'filter_configs': args.filter_configs,
        'epochs': args.epochs,
        'patience': args.patience,
        'learning_rate': args.learning_rate,
        'seed': args.seed,
        'timestamp': timestamp
    }

    with open(os.path.join(output_dir, 'config_summary.json'), 'w') as f:
        json.dump(config_summary, f, indent=4)

    # Set device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    try:
        # Create datasets with appropriate transforms
        train_val_dataset = MicrobeadDataset(image_dir, density_file, transform=train_transform)
        test_dataset = MicrobeadDataset(image_dir, density_file, transform=eval_transform)

        # Split dataset into train, validation, and test sets
        total_size = len(train_val_dataset)
        train_size = int(0.7 * total_size)
        val_size = int(0.15 * total_size)
        test_size = total_size - train_size - val_size

        # Create indices for the splits
        indices = list(range(total_size))
        np.random.shuffle(indices)

        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size+val_size]
        test_indices = indices[train_size+val_size:]

        # Create subset samplers
        from torch.utils.data import SubsetRandomSampler
        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)
        test_sampler = SubsetRandomSampler(test_indices)

        print(f"Dataset split: {train_size} training, {val_size} validation, {test_size} test samples")

        # Experiment result tracker
        all_results = []

        # Create combinations of batch size and filter configurations
        for batch_size in args.batch_sizes:
            for filter_str in args.filter_configs:
                # Parse filter configuration
                filter_config = [int(x) for x in filter_str.split(',')]

                print(f"\n{'='*80}")
                print(f"STARTING EXPERIMENT: Batch Size = {batch_size}, Filters = {filter_config}")
                print(f"{'='*80}\n")

                # Configure experiment
                experiment_config = {
                    'batch_size': batch_size,
                    'filter_config': filter_config,
                    'num_epochs': args.epochs,
                    'patience': args.patience,
                    'learning_rate': args.learning_rate,
                    'output_dir': output_dir
                }

                # Create data loaders for this batch size
                train_loader = DataLoader(train_val_dataset, batch_size=batch_size,
                                         sampler=train_sampler, num_workers=4)
                val_loader = DataLoader(train_val_dataset, batch_size=batch_size,
                                       sampler=val_sampler, num_workers=4)
                test_loader = DataLoader(test_dataset, batch_size=batch_size,
                                        sampler=test_sampler, num_workers=4)

                # Initialize model with filter configuration
                model = DensityRegressionCNN(filters=filter_config)
                print(f"Model initialized with filters: {filter_config}")

                # Train the model
                print("\nStarting training...")
                trained_model, training_stats, epochs_completed = train_model(
                    model,
                    train_loader,
                    val_loader,
                    experiment_config,
                    device=device
                )

                # Evaluate the model
                print("\nEvaluating model performance...")
                eval_metrics = evaluate_model(trained_model, test_loader, experiment_config, device)

                # Combine results
                experiment_results = {
                    'config': {
                        'batch_size': batch_size,
                        'filter_config': filter_config
                    },
                    'training': training_stats,
                    'evaluation': eval_metrics
                }

                all_results.append(experiment_results)

                # Save individual experiment results
                experiment_file = f"results_batch{batch_size}_filters{'-'.join(map(str, filter_config))}.json"
                with open(os.path.join(output_dir, experiment_file), 'w') as f:
                    json.dump(experiment_results, f, indent=4)

        # Save all results in one file
        with open(os.path.join(output_dir, 'all_experiment_results.json'), 'w') as f:
            json.dump(all_results, f, indent=4)

        # Create comparison table
        comparison_data = []
        for result in all_results:
            comparison_data.append({
                'Batch Size': result['config']['batch_size'],
                'Filter Config': '-'.join(map(str, result['config']['filter_config'])),
                'Best Val Loss': result['training']['performance']['best_val_loss'],
                'MSE': result['evaluation']['mse'],
                'MAE': result['evaluation']['mae'],
                'R²': result['evaluation']['r2'],
                'Epochs': result['training']['performance']['epochs_completed'],
                'Training Time (min)': result['training']['performance']['training_minutes']
            })

        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('MSE')  # Sort by MSE (best to worst)
        comparison_df.to_csv(os.path.join(output_dir, 'experiment_comparison.csv'), index=False)

        # Display best configuration
        best_config = comparison_df.iloc[0]
        print("\n" + "="*80)
        print("BEST CONFIGURATION:")
        print(f"Batch Size: {best_config['Batch Size']}")
        print(f"Filter Configuration: {best_config['Filter Config']}")
        print(f"MSE: {best_config['MSE']:.4f}, MAE: {best_config['MAE']:.4f}, R²: {best_config['R²']:.4f}")
        print(f"Training Time: {best_config['Training Time (min)']:.2f} minutes")
        print("="*80)

        print(f"\nAll experiment results saved to: {output_dir}")

    except Exception as e:
        print(f"Error during execution: {str(e)}")
        import traceback
        traceback.print_exc()
