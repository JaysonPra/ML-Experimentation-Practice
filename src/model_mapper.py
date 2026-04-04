from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

def _scale_map(scale_ftrs, ignore_ftrs):
    mapper = {
        "standard": StandardScaler(),
        "minmax": MinMaxScaler(),
        "robust": RobustScaler(),
    }

    transformer = []

    for scaler, feature in scale_ftrs.items():
        active_ftrs = [col for col in feature if col not in ignore_ftrs]
        if active_ftrs:
            scaler_object = mapper.get(scaler.lower())
            if scaler_object:
                transformer.append((scaler, scaler_object, active_ftrs))

    return transformer