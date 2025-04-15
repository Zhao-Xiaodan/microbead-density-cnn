
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

# 1. Define the CNN Model
class DensityRegressionCNN(nn.Module):
    def __init__(self):
        super(DensityRegressionCNN, self).__init__()
        # Convolutional layers for feature extraction
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # Input: 1 channel (grayscale)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)  # 2x2 pooling
        self.bn1 = nn.BatchNorm2d(32)   # Batch normalization for stability
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU()
        # Fully connected layers for regression
        self.fc1 = nn.Linear(128, 64)   # After global pooling
        self.fc2 = nn.Linear(64, 1)     # Output: single density value
        self.dropout = nn.Dropout(0.5)  # Prevent overfitting

    def forward(self, x):
        # Convolutional stack with pooling
        x = self.pool(self.relu(self.bn1(self.conv1(x))))  # 224x224 → 112x112
        x = self.pool(self.relu(self.bn2(self.conv2(x))))  # 112x112 → 56x56
        x = self.pool(self.relu(self.bn3(self.conv3(x))))  # 56x56 → 28x28
        # Global Average Pooling
        x = x.mean(dim=[2, 3])  # Reduce 28x28x128 → 128
        # Fully connected layers
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)  # No activation (linear output for regression)
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
    transforms.Resize((224, 224)),         # Resize to 224x224
    transforms.ToTensor(),                 # Convert to tensor (0-1 range)
    transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalize to [-1, 1]
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
def train_model(model, train_loader, val_loader, num_epochs=40, patience=10, device='cuda'):
    criterion = nn.MSELoss()  # Mean Squared Error for regression
    optimizer = optim.Adam(model.parameters(), lr=0.001)
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

    start_time = time.time()

    for epoch in range(num_epochs):
        epochs_completed = epoch + 1

        # Training phase
        model.train()
        train_loss = 0.0
        for images, densities in train_loader:
            images, densities = images.to(device), densities.to(device)
            optimizer.zero_grad()
            outputs = model(images).squeeze()  # Remove extra dim
            loss = criterion(outputs, densities)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
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
            torch.save(model.state_dict(), 'training_results/best_model.pth')
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
    print(f'Training completed in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s')
    print(f'Best validation loss: {best_val_loss:.4f}')

    # Load the best model weights
    model.load_state_dict(best_model_wts)

    # Create results directory if it doesn't exist
    os.makedirs('training_results', exist_ok=True)

    # Plot training curves
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.title(f'Training and Validation Loss (Completed {epochs_completed} epochs)')
    plt.legend()
    plt.savefig('training_results/training_curve.png')
    plt.close()

    return model, history, epochs_completed

# Function to evaluate model performance
def evaluate_model(model, test_loader, device):
    model.eval()
    predictions = []
    actual_values = []
    filenames = []

    with torch.no_grad():
        for images, densities in test_loader:
            images, densities = images.to(device), densities.to(device)
            outputs = model(images).squeeze()

            predictions.extend(outputs.cpu().numpy())
            actual_values.extend(densities.cpu().numpy())

            # Get filenames for the current batch
            for i in range(len(images)):
                idx = test_loader.dataset.indices[len(predictions) - len(images) + i]
                filename = test_loader.dataset.dataset.df.iloc[idx]['filename']
                filenames.append(filename)

    # Convert to numpy arrays
    predictions = np.array(predictions)
    actual_values = np.array(actual_values)

    # Calculate metrics
    mse = np.mean((predictions - actual_values) ** 2)
    mae = np.mean(np.abs(predictions - actual_values))
    r2 = 1 - (np.sum((actual_values - predictions) ** 2) /
              np.sum((actual_values - np.mean(actual_values)) ** 2))

    print(f"Model Evaluation Metrics:")
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
    plt.title('Predicted vs Actual Density')
    plt.savefig('training_results/prediction_performance.png')
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
    results_df.to_csv('training_results/test_predictions.csv', index=False)
    print(f"Test predictions saved to 'training_results/test_predictions.csv'")

    return mse, mae, r2

# Main Execution
if __name__ == "__main__":
    # Set paths
    base_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory of this script
    image_dir = os.path.join(base_dir, 'dataset', 'images')
    density_file = os.path.join(base_dir, 'dataset', 'density.csv')

    # Create results directory
    os.makedirs('training_results', exist_ok=True)

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

        # Data loaders
        train_loader = DataLoader(train_val_dataset, batch_size=32, sampler=train_sampler, num_workers=4)
        val_loader = DataLoader(train_val_dataset, batch_size=32, sampler=val_sampler, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=32, sampler=test_sampler, num_workers=4)

        print(f"Dataset split: {train_size} training, {val_size} validation, {test_size} test samples")

        # Initialize model
        model = DensityRegressionCNN()
        print("Model initialized:")
        print(model)

        # Train the model with early stopping and extended epochs
        print("\nStarting training...")
        trained_model, history, epochs_completed = train_model(
            model,
            train_loader,
            val_loader,
            num_epochs=40,  # Increased from 20 to 40
            patience=10,    # Early stopping after 10 epochs without improvement
            device=device
        )

        # Evaluate the model
        print("\nEvaluating model performance...")
        evaluate_model(trained_model, test_loader, device)

        # Save the final model (best model was already saved during training)
        torch.save(trained_model.state_dict(), 'training_results/microbead_density_model_final.pth')
        print("\nFinal model saved as 'training_results/microbead_density_model_final.pth'")
        print(f"Best model saved as 'training_results/best_model.pth'")

        # Save training history
        history_df = pd.DataFrame({
            'epoch': list(range(1, epochs_completed + 1)),
            'train_loss': history['train_loss'],
            'val_loss': history['val_loss']
        })
        history_df.to_csv('training_results/training_history.csv', index=False)
        print(f"Training history saved to 'training_results/training_history.csv'")

    except Exception as e:
        print(f"Error during execution: {str(e)}")
        import traceback
        traceback.print_exc()
