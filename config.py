from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent

DATA_DIR = ROOT_DIR / "data"
EXPERIMENTATION_DIR = ROOT_DIR / "experiments"

DATASET_HANDLE = "borovai0/student-performance-analytics-dataset"
DATASET_ZIP_NAME = DATASET_HANDLE.split('/')[-1]
DATASET_CSV_NAME = "student_performance_data"