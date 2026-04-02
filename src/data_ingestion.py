from kaggle.api.kaggle_api_extended import KaggleApi
from config import DATA_DIR, DATASET_HANDLE
import zipfile
import os

def run_ingestion(save_location=DATA_DIR):
    api = KaggleApi()
    api.authenticate()

    dataset_handle = DATASET_HANDLE
    file_name = dataset_handle.split('/')[-1]
    print(file_name)
    zip_path = save_location / f"{file_name}.zip"
    print(zip_path)

    api.dataset_download_files(
        dataset="borovai0/student-performance-analytics-dataset",
        path=save_location
    )

    if os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path, 'r') as zipped_file:
            zipped_file.extractall(path=save_location)
            os.remove(zip_path)