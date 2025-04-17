#!/bin/bash
#PBS -l walltime=300:00:00
#PBS -l select=1:mem=20g:ncpus=8:ngpus=1
#PBS -N Microbead_CNN
#PBS -o Microbead_CNN.out
#PBS -j oe
#PBS -M phyzxi@nus.edu.sg
#PBS -m abe
#PBS -P yanjie

# Change to the directory where the job was submitted
cd ~/CNN/

# Activate conda environment - IMPORTANT
source ~/.bashrc
conda activate microbead-cnn  # Replace with your environment name

# Set environment variables for better PyTorch performance
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# Set CUDA visible devices if needed
export CUDA_VISIBLE_DEVICES=0

# Log some information about the job
echo "Job started on $(date)"
echo "Running on node: $(hostname)"
echo "Using GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Python executable: $(which python)"

# Run the Python script with experimental configurations
python train_microbead_cnn_v3.py \
  --batch_sizes 16 32 64 \
  --filter_configs "32,64,128" "16,32,64" "64,128,256" "128,256,512" \
  --epochs 60 \
  --patience 15 \
  --learning_rate 0.001 \
  --output_dir training_results

echo "Job finished on $(date)"
