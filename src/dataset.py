"""
dataset.py — Kaggle Dataset Download & Setup Module

This module handles the automatic download of the Retailrocket eCommerce
dataset from Kaggle using the Kaggle API.  It checks whether the data
already exists locally and only downloads if needed.

Prerequisites:
    - The 'kaggle' Python package must be installed.
    - A valid Kaggle API key (kaggle.json) must be configured at ~/.kaggle/.

Key Exports:
    - setup_data():  Ensures the dataset directory exists and contains
                     the required CSV files.  Downloads from Kaggle if missing.
"""

# =============================================================================
# Imports
# =============================================================================

import kaggle
import os

# =============================================================================
# Constants
# =============================================================================

# Kaggle dataset identifier (owner/dataset-name)
DATASET_NAME = "retailrocket/ecommerce-dataset"

# Local directory where dataset CSV files will be stored
TARGET_DIR = "./dataset"

# =============================================================================
# Public Functions
# =============================================================================


def setup_data():
    """
    Ensure the Retailrocket dataset is available locally.

    This function performs two checks:
      1. Creates the target directory if it doesn't exist.
      2. Checks if any .csv files are already present in the directory.
         - If CSV files exist: prints a message and returns (no download).
         - If no CSV files: downloads the dataset from Kaggle, extracts
           the zip archive, and places CSV files in the target directory.

    The Kaggle API is used for downloading.  If the download fails (e.g.,
    missing API key, network error), the exception is caught and printed.

    Returns:
        None
    """
    # Create the dataset directory if it doesn't exist yet
    if not os.path.exists(TARGET_DIR):
        os.makedirs(TARGET_DIR)
        print(f"Create directory {TARGET_DIR}")

    # Check if CSV files are already present (skip download if so)
    files_in_dir = os.listdir(TARGET_DIR)
    if not any(f.endswith('.csv') for f in files_in_dir):
        print(f"Downloading dataset to '{TARGET_DIR}'...")
        try:
            # Download and automatically unzip the dataset
            kaggle.api.dataset_download_files(
                DATASET_NAME, path=TARGET_DIR, unzip=True
            )
            print("Download completed successfully!")
        except Exception as e:
            print(f"Error occurred while downloading: {e}")
    else:
        print("Data is already available in the 'dataset' folder.")