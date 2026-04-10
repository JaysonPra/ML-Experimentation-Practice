import mlflow
import optuna
from optuna.trial import Trial
import yaml
import pandas as pd
from optuna.visualization import plot_pareto_front
import argparse
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

from src.model_mapper import scale_map
from src.model_train import train_model
from src.pipeline import build_pipeline
from config import EXPERIMENTATION_DIR

def _suggest_param(trial: Trial, name, value):
    if isinstance(value, list):
        if len(value) == 2 and isinstance(value[0], int) and not isinstance(value[0], bool):
            return trial.suggest_int(name, value[0], value[1])

        if len(value) == 2 and isinstance(value[0], float):
            return trial.suggest_float(name, value[0], value[1])
        
        return trial.suggest_categorical(name, value)
    
    return value

def objective(trial: Trial, df, experiment_file, search_grid, target_col):
    with mlflow.start_run(nested=True) as run:
        trial.set_user_attr("mlflow_run_id", run.info.run_id)

        model_idx = trial.suggest_int("model_idx", 0, len(search_grid) - 1)
        model_cfg = search_grid[model_idx]
        model_class = model_cfg["regressor"]
        model_name = model_class.__name__

        scaler_map = {
            "StandardScaler": StandardScaler,
            "RobustScaler": RobustScaler,
            "MinMaxScaler": MinMaxScaler,
            "None": None
        }

        available_scaler_names = [
            s.__name__ if hasattr(s, '__name__') else "None" 
            for s in model_cfg["scalers"]
        ]

        scaler_name = trial.suggest_categorical(f"{model_name}_scaler", available_scaler_names)

        scaler_class = scaler_map[scaler_name]
        num_scaler = scaler_class() if scaler_class is not None else None

        params = {}
        for key, value in model_cfg.items():
            if key in ["regressor", "scalers"]:
                continue
            p_name = f"{model_name}_{key}"
            params[key] = _suggest_param(trial, p_name, value)

        model_instance = model_class(**params)

        n_splits = experiment_file["train"]["n_splits"]
        rmse, latency = train_model(n_splits, num_scaler, model_instance, target_col, df)

        mlflow.log_metric("Average RMSE", rmse)
        mlflow.log_metric("Average Latency", latency)
    
        final_trial_pipeline = build_pipeline(df, num_scaler, model_instance, target_col)
        final_trial_pipeline.fit(df.drop(columns=[target_col]), df[target_col])
        mlflow.sklearn.log_model(
            sk_model=final_trial_pipeline,
            artifact_path="model"
        )

        return rmse, latency
        
def start_experiment(experiment_file_name):
    with open(experiment_file_name, 'r') as yaml_file:
        experiment_file = yaml.safe_load(yaml_file)
    df = pd.read_csv(experiment_file["data"]["file_name"])

    mlflow.set_experiment(experiment_file["experiment_name"])

    search_grid = scale_map(experiment_file["train"]["search_grid"])
    target_col = experiment_file["data"]["target_col"]

    with mlflow.start_run(run_name="Optimization_Summary") as parent_run:
        study = optuna.create_study(directions=["minimize", "minimize"])
        study.optimize(
            lambda trial: objective(trial, df, experiment_file, search_grid, target_col), 
            n_trials=experiment_file["train"]["n_trials"]
        )

        pareto_front = plot_pareto_front(study, target_names=["RMSE", "Latency"])
        mlflow.log_figure(pareto_front, "pareto_front.html")

        mlflow.set_tag("best_trials_count", len(study.best_trials))

    for trial in study.best_trials:
        with mlflow.start_run(run_id=trial.user_attrs.get("mlflow_run_id")):
            mlflow.set_tag("is_pareto", "True")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Regression")
    
    parser.add_argument(
        "--file",
        required=True,
        type=str,
        help="Write the name of the YAML experimentation file"
    )

    args = parser.parse_args()
    if args.file:
        experiment_file = EXPERIMENTATION_DIR / args.file
        start_experiment(experiment_file)