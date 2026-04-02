from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent

DATA_DIR = ROOT_DIR / "data"
EXPERIMENTATION_DIR = ROOT_DIR / "experiments"

DATASET_HANDLE = "borovai0/student-performance-analytics-dataset"
DATASET_NAME = DATASET_HANDLE.split('/')[-1]