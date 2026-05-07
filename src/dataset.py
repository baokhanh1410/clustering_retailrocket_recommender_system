import kaggle
import os
dataset_name = "retailrocket/ecommerce-dataset"
target_dir = "./dataset"

def setup_data():
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print(f"Create directory {target_dir}")
    files_in_dir = os.listdir(target_dir)
    if not any(f.endswith('.csv') for f in files_in_dir):
        print(f"Downloading dataset to '{target_dir}'...")
        try:
            kaggle.api.dataset_download_files(dataset_name, path=target_dir, unzip=True)
            print("Download completed successfully!")
        except Exception as e:
            print(f"Error occurred while downloading: {e}")
    else:
        print("Data is already available in the 'dataset' folder.")