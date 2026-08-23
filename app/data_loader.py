from pathlib import Path
import pandas as pd
import joblib


# Routes
APP_DIR = Path(__file__).parent
PROJECT_ROOT = APP_DIR.parent

DATA_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"

# Dataset
df = pd.read_csv(DATA_DIR / "spotify_clean.csv")


# Model
model = joblib.load(
    MODELS_DIR / "random_forest_pipeline.joblib"
)


# Notebook 4 results
model_results = pd.read_csv(
    RESULTS_DIR / "model_results.csv"
)

final_metrics = pd.read_csv(
    RESULTS_DIR / "final_metrics.csv"
)

predictions = pd.read_csv(
    RESULTS_DIR / "predictions.csv"
)

feature_importance = pd.read_csv(
    RESULTS_DIR / "feature_importance.csv"
)


# Notebook 5 results
pca_variance = pd.read_csv(
    RESULTS_DIR / "pca_variance.csv"
)

pca_coordinates = pd.read_csv(
    RESULTS_DIR / "pca_coordinates.csv"
)

pca_correlations = pd.read_csv(
    RESULTS_DIR / "pca_correlations.csv"
)

clustering_metrics = pd.read_csv(
    RESULTS_DIR / "clustering_metrics.csv"
)

cluster_profiles = pd.read_csv(
    RESULTS_DIR / "cluster_profiles.csv"
)

clustered_data = pd.read_csv(
    RESULTS_DIR / "clustered_data.csv"
)

popularity_by_cluster = pd.read_csv(
    RESULTS_DIR / "popularity_by_cluster.csv"
)