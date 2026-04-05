from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

def scale_map(search_grid):
    regressor_mapper = {
        "linear_regression": LinearRegression,
        "random_forest": RandomForestRegressor,
        "xgboost": XGBRegressor,
    }
    
    scaler_mapper = {
        "standard": StandardScaler,
        "minmax": MinMaxScaler,
        "robust": RobustScaler,
    }

    for model_cfg in search_grid:
        reg_name = model_cfg["regressor"][0]
        model_cfg["regressor"] = regressor_mapper.get(reg_name)
        
        scaler_names = model_cfg.get("scalers", [])
        model_cfg["scalers"] = [scaler_mapper.get(s) for s in scaler_names]

    return search_grid