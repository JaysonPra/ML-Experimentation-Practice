from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
import numpy as np

def build_pipeline(df, num_scaler, model, target_col):
    features = df.drop(columns=[target_col])

    object_cols = features.select_dtypes(include=['object']).columns.tolist()
    num_cols = features.select_dtypes(include=[np.number]).columns.tolist()
    
    bin_cols = [c for c in object_cols if features[c].nunique() == 2]
    nom_cols = [c for c in object_cols if features[c].nunique() > 2]
    
    scaler_to_use = num_scaler if num_scaler is not None else "passthrough"

    categorical_encoder = ColumnTransformer([
        ("ordinal", OrdinalEncoder(), bin_cols),
        ("nominal", OneHotEncoder(handle_unknown='ignore'), nom_cols),
        ("numerical", scaler_to_use, num_cols)
    ], remainder='passthrough')

    pipeline = Pipeline([
        ("preprocessor",categorical_encoder),
        ("model", model)
    ])

    return pipeline