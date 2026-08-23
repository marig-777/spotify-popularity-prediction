from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold, cross_validate
import tensorflow as tf
import keras
from keras import layers

def build_pipeline(model, preprocessor) -> Pipeline:
    pipeline = Pipeline (
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )
    
    return pipeline

def train_and_evaluate(pipeline, X_train, y_train, X_test, y_test):
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return {
        "mae" : mae,
        "rmse" : rmse,
        "r2" : r2
    }

def cross_validate_model(pipeline, X_train, y_train, cv = 5):
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    scoring = {
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
        "r2": "r2"
    }

    results = cross_validate(pipeline, X_train, y_train, cv=kf, scoring=scoring)

    mae = -results["test_mae"]
    rmse = -results["test_rmse"]
    r2 = results["test_r2"]

    return {
        "mae_mean": mae.mean(),
        "mae_std": mae.std(),
        "rmse_mean": rmse.mean(),
        "rmse_std": rmse.std(),
        "r2_mean": r2.mean(),
        "r2_std": r2.std()
    }

def create_deep_learning_model(X, y=None):
    input_dim = X.shape[1]

    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(256, activation="relu"),
        layers.Dense(128, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(32, activation="relu"),
        layers.Dense(1)
    ])

    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"]
    )
    return model


def evaluate_models(models, X_train, y_train, cv=5):
    results = {}
    for name, pipeline in models.items():
        results[name] = cross_validate_model(
            pipeline,
            X_train,
            y_train,
            cv
        )
    
    return results
