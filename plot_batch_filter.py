
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import sys
import argparse
from pathlib import Path
import glob

def find_experiment_csv_files():
    """Find all experiment_comparison.csv files in current directory and subdirectories"""
    target_filename = "experiment_comparison.csv"

    # Search for files recursively
    experiment_files = list(Path('.').rglob(target_filename))

    return experiment_files

def get_csv_path():
    """Get the CSV file path from user input or command line argument"""

    # Check for command line argument
    parser = argparse.ArgumentParser(description='Analyze experiment comparison results')
    parser.add_argument('--input_dir', type=str, help='Path to experiment_comparison.csv file')
    args, unknown = parser.parse_known_args()

    if args.input_dir:
        if os.path.exists(args.input_dir):
            return args.input_dir
        else:
            print(f"Error: File not found at {args.input_dir}")
            return None

    # No path provided, search for available files
    print("No path to experiment_comparison.csv provided.")
    print("Available CSV files in current directory and subdirectories:")

    experiment_files = find_experiment_csv_files()

    if not experiment_files:
        print(f"No 'experiment_comparison.csv' files found in current directory and subdirectories.")
        return None

    print(f"Found {len(experiment_files)} 'experiment_comparison.csv' files:")
    for i, file_path in enumerate(experiment_files):
        # Show relative path for clarity
        rel_path = file_path.relative_to(Path('.'))
        print(f"{i+1}. {rel_path}")

    # Let user select from available files
    try:
        selection = int(input("Enter the number of the file you want to use (or 0 to exit): "))
        if selection == 0:
            print("Exiting program.")
            return None
        if 1 <= selection <= len(experiment_files):
            return str(experiment_files[selection-1])
        else:
            print("Invalid selection. Exiting program.")
            return None
    except ValueError:
        print("Invalid input. Exiting program.")
        return None

def analyze_experiments(csv_path):
    """Main analysis function"""

    # Load the experiment comparison data
    try:
        df = pd.read_csv(csv_path)
        print(f"Successfully loaded data from {csv_path}")
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return

    # Validate required columns
    required_columns = ['Batch Size', 'Filter Config', 'MSE', 'MAE', 'R²', 'Best Val Loss', 'Training Time (min)', 'Epochs']
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        print(f"Error: Missing required columns in CSV file: {', '.join(missing_columns)}")
        print(f"Available columns: {', '.join(df.columns)}")
        return

    # Display the results table sorted by MSE (best to worst)
    print("\nModel Comparison Results (Sorted by MSE):")
    print(df.sort_values('MSE').to_string(index=False))

    # Create a figure with multiple subplots
    plt.figure(figsize=(18, 15))

    # 1. Plot MSE, MAE, and R² for each configuration
    plt.subplot(2, 2, 1)
    bar_width = 0.25
    index = np.arange(len(df))
    labels = [f"B{row['Batch Size']}_F{row['Filter Config']}" for _, row in df.iterrows()]

    plt.bar(index - bar_width, df['MSE'], width=bar_width, label='MSE')
    plt.bar(index, df['MAE'], width=bar_width, label='MAE')
    plt.bar(index + bar_width, 1 - df['R²'], width=bar_width, label='1-R²')  # Using 1-R² to keep scale similar
    plt.yscale('log')
    plt.xlabel('Model Configuration')
    plt.ylabel('Error Metrics')
    plt.title('Error Metrics by Model Configuration')
    plt.xticks(index, labels, rotation=45)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # 2. Plot a comparison of validation loss vs. test MSE
    plt.subplot(2, 2, 2)
    plt.scatter(df['Best Val Loss'], df['MSE'], s=100, alpha=0.7)
    for i, row in df.iterrows():
        plt.annotate(f"B{row['Batch Size']}_F{row['Filter Config']}",
                    (row['Best Val Loss'], row['MSE']),
                    xytext=(5, 5), textcoords='offset points')
    plt.xlabel('Best Validation Loss')
    plt.ylabel('Test MSE')
    plt.title('Validation Loss vs Test MSE')
    plt.grid(True, linestyle='--', alpha=0.7)

    # 3. Plot training time vs. MSE
    plt.subplot(2, 2, 3)
    plt.scatter(df['Training Time (min)'], df['MSE'], s=100, alpha=0.7)
    for i, row in df.iterrows():
        plt.annotate(f"B{row['Batch Size']}_F{row['Filter Config']}",
                    (row['Training Time (min)'], row['MSE']),
                    xytext=(5, 5), textcoords='offset points')
    plt.xlabel('Training Time (minutes)')
    plt.ylabel('Test MSE')
    plt.title('Training Time vs Test MSE')
    plt.grid(True, linestyle='--', alpha=0.7)

    # 4. Plot Batch Size vs. Filter Config with MSE as color
    plt.subplot(2, 2, 4)

    try:
        # Create a pivot table for the heatmap
        pivot = df.pivot_table(
            index='Batch Size',
            columns='Filter Config',
            values='MSE'
        )

        sns.heatmap(pivot, annot=True, fmt=".4f", cmap="YlGnBu_r", cbar_kws={'label': 'MSE (lower is better)'})
        plt.title('MSE by Batch Size and Filter Configuration')
    except ValueError as e:
        plt.text(0.5, 0.5, f"Could not create heatmap: {e}",
                 horizontalalignment='center', verticalalignment='center')
        plt.title('Heatmap Error')

    plt.tight_layout()

    # Create a parallel coordinates plot for all parameters
    plt.figure(figsize=(15, 8))

    # Normalize the data for parallel coordinates
    norm_df = df.copy()
    for column in ['Batch Size', 'Best Val Loss', 'MSE', 'MAE', 'Training Time (min)']:
        if norm_df[column].max() > norm_df[column].min():
            norm_df[column] = (norm_df[column] - norm_df[column].min()) / (norm_df[column].max() - norm_df[column].min())

    # Drop R² and Filter Config for this plot
    plot_cols = ['Batch Size', 'Best Val Loss', 'MSE', 'MAE', 'Training Time (min)', 'Epochs']
    parallel_df = norm_df[plot_cols]

    # Plot parallel coordinates
    pd.plotting.parallel_coordinates(
        parallel_df, 'Batch Size',
        color=plt.cm.viridis(np.linspace(0, 1, len(df))),
        linewidth=2,
        alpha=0.7
    )
    plt.title('Parallel Coordinates Plot of Model Parameters')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend([f"B{row['Batch Size']}_F{row['Filter Config']}" for _, row in df.iterrows()],
               loc='upper right', bbox_to_anchor=(1.15, 1))

    plt.tight_layout()

    # Create output directory if it doesn't exist
    output_dir = 'analysis_results'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_path = os.path.join(output_dir, 'model_comparison_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved visualization to {output_path}")
    plt.show()

    # Identify best models by different metrics
    best_mse = df.loc[df['MSE'].idxmin()]
    best_mae = df.loc[df['MAE'].idxmin()]
    best_r2 = df.loc[df['R²'].idxmax()]
    fastest = df.loc[df['Training Time (min)'].idxmin()]

    print("\n===== Best Models by Different Metrics =====")
    print(f"\nBest MSE Model:")
    print(f"  Batch Size: {best_mse['Batch Size']}, Filter Config: {best_mse['Filter Config']}")
    print(f"  MSE: {best_mse['MSE']:.4f}, MAE: {best_mse['MAE']:.4f}, R²: {best_mse['R²']:.4f}")
    print(f"  Training Time: {best_mse['Training Time (min)']:.2f} minutes")

    print(f"\nBest MAE Model:")
    print(f"  Batch Size: {best_mae['Batch Size']}, Filter Config: {best_mae['Filter Config']}")
    print(f"  MSE: {best_mae['MSE']:.4f}, MAE: {best_mae['MAE']:.4f}, R²: {best_mae['R²']:.4f}")
    print(f"  Training Time: {best_mae['Training Time (min)']:.2f} minutes")

    print(f"\nBest R² Model:")
    print(f"  Batch Size: {best_r2['Batch Size']}, Filter Config: {best_r2['Filter Config']}")
    print(f"  MSE: {best_r2['MSE']:.4f}, MAE: {best_r2['MAE']:.4f}, R²: {best_r2['R²']:.4f}")
    print(f"  Training Time: {best_r2['Training Time (min)']:.2f} minutes")

    print(f"\nFastest Model:")
    print(f"  Batch Size: {fastest['Batch Size']}, Filter Config: {fastest['Filter Config']}")
    print(f"  MSE: {fastest['MSE']:.4f}, MAE: {fastest['MAE']:.4f}, R²: {fastest['R²']:.4f}")
    print(f"  Training Time: {fastest['Training Time (min)']:.2f} minutes")

    # Calculate tradeoff score (lower is better)
    # Normalize metrics between 0-1
    min_mse, max_mse = df['MSE'].min(), df['MSE'].max()
    min_time, max_time = df['Training Time (min)'].min(), df['Training Time (min)'].max()

    df['MSE_norm'] = (df['MSE'] - min_mse) / (max_mse - min_mse) if max_mse > min_mse else 0
    df['Time_norm'] = (df['Training Time (min)'] - min_time) / (max_time - min_time) if max_time > min_time else 0

    # Calculate balanced score (70% accuracy, 30% speed)
    df['Tradeoff_Score'] = 0.7 * df['MSE_norm'] + 0.3 * df['Time_norm']
    best_tradeoff = df.loc[df['Tradeoff_Score'].idxmin()]

    print(f"\nBest Tradeoff Model (70% accuracy, 30% speed):")
    print(f"  Batch Size: {best_tradeoff['Batch Size']}, Filter Config: {best_tradeoff['Filter Config']}")
    print(f"  MSE: {best_tradeoff['MSE']:.4f}, MAE: {best_tradeoff['MAE']:.4f}, R²: {best_tradeoff['R²']:.4f}")
    print(f"  Training Time: {best_tradeoff['Training Time (min)']:.2f} minutes")

    # Final recommendation
    print("\n===== Model Recommendation =====")
    print("Based on overall performance, I recommend using the model with:")
    print(f"  Batch Size: {best_mse['Batch Size']}, Filter Config: {best_mse['Filter Config']}")
    print(f"  MSE: {best_mse['MSE']:.4f}, MAE: {best_mse['MAE']:.4f}, R²: {best_mse['R²']:.4f}")
    print(f"  Training Time: {best_mse['Training Time (min)']:.2f} minutes")
    print("\nRationale:")
    print("1. This model achieves the lowest MSE, which is typically the primary metric for regression tasks")
    print("2. The R² value indicates strong predictive power")
    print("3. The training time is reasonable for the performance gained")

    if best_tradeoff['Batch Size'] != best_mse['Batch Size'] or best_tradeoff['Filter Config'] != best_mse['Filter Config']:
        print("\nAlternative recommendation for better efficiency:")
        print(f"  Batch Size: {best_tradeoff['Batch Size']}, Filter Config: {best_tradeoff['Filter Config']}")
        print(f"  MSE: {best_tradeoff['MSE']:.4f}, MAE: {best_tradeoff['MAE']:.4f}, R²: {best_tradeoff['R²']:.4f}")
        print(f"  Training Time: {best_tradeoff['Training Time (min)']:.2f} minutes")
        print("  This model provides a good balance between accuracy and computational efficiency")

    # Save results to CSV
    df_sorted = df.sort_values('MSE')
    results_path = os.path.join(output_dir, 'model_comparison_results.csv')
    df_sorted.to_csv(results_path, index=False)
    print(f"\nSaved detailed results to {results_path}")

if __name__ == "__main__":
    csv_path = get_csv_path()
    if csv_path:
        analyze_experiments(csv_path)
