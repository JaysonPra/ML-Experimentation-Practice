from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
import numpy as np

def build_pipeline(df, num_scaler, model_type, params):
    object_cols = df.select_dtypes(include=['object']).columns.tolist()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    bin_cols = [c for c in object_cols if df[c].nunique() == 2]
    nom_cols = [c for c in object_cols if df[c].nunique() > 2]
    
    categorical_encoder = ColumnTransformer([
        ("ordinal", OrdinalEncoder(), bin_cols),
        ("nominal", OneHotEncoder(handle_unknown='ignore'), nom_cols),
        ("numerical", num_scaler, num_cols)
    ])

    pipeline = Pipeline([
        ("preprocessor",categorical_encoder),
        ("model", model_type(**params))
    ])

    return pipeline