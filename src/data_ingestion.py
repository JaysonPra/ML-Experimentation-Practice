from kaggle.api.kaggle_api_extended import KaggleApi
from config import DATA_DIR, DATASET_HANDLE, DATASET_ZIP_NAME
import zipfile
import os

def run_ingestion(save_location=DATA_DIR):
    api = KaggleApi()
    api.authenticate()

    dataset_handle = DATASET_HANDLE
    file_name = DATASET_ZIP_NAME
    zip_path = save_location / f"{file_name}.zip"

    api.dataset_download_files(
        dataset=dataset_handle,
        path=save_location
    )

    if os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path, 'r') as zipped_file:
            zipped_file.extractall(path=save_location)
            os.remove(zip_path)