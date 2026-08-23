import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_histogram(data: pd.DataFrame, variable: str):
    """Histograma de una variable numerica"""
    fig, ax = plt.subplots(figsize=(8, 5))

    values = data[variable].dropna()

    ax.hist(values, bins=30, edgecolor="black")
    ax.set_title(f"Distribución de {variable}")
    ax.set_xlabel(variable)
    ax.set_ylabel("Frecuencia")

    fig.tight_layout()
    return fig


def plot_boxplot(data: pd.DataFrame, variable: str):
    """Boxplot de una variable numérica"""
    fig, ax = plt.subplots(figsize=(8, 4))

    values = data[variable].dropna()

    ax.boxplot(values, vert=False)
    ax.set_title(f"Distribución de {variable}")
    ax.set_xlabel(variable)

    fig.tight_layout()
    return fig


def plot_correlation_heatmap(data: pd.DataFrame):
    """Mapa de correlaciones para las variables numéricas"""
    numeric_data = data.select_dtypes(include="number")
    correlation = numeric_data.corr()

    fig, ax = plt.subplots(figsize=(10, 8))

    image = ax.imshow(
        correlation,
        aspect="auto",
        vmin=-1,
        vmax=1,
    )

    ax.set_xticks(range(len(correlation.columns)))
    ax.set_yticks(range(len(correlation.columns)))

    ax.set_xticklabels(
        correlation.columns,
        rotation=45,
        ha="right",
    )

    ax.set_yticklabels(correlation.columns)
    ax.set_title("Matriz de correlaciones")

    fig.colorbar(image, ax=ax, label="Correlación")
    fig.tight_layout()

    return fig


def plot_model_comparison(model_results: pd.DataFrame):
    """R^2 promedio obtenido por cada modelo"""
    required_columns = {"modelo", "r2_mean"}

    if not required_columns.issubset(model_results.columns):
        raise ValueError(
            "model_results debe incluir las columnas "
            "'modelo' y 'r2_mean'"
        )

    ordered = model_results.sort_values("r2_mean")

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.barh(
        ordered["modelo"],
        ordered["r2_mean"],
    )

    ax.set_title("Comparación de modelos")
    ax.set_xlabel("R^2 promedio")
    ax.set_ylabel("Modelo")

    fig.tight_layout()
    return fig


def plot_feature_importance(feature_importance: pd.DataFrame, top_n: int = 15,):
    """Muestra las características con mayor importancia"""
    variable_col = next(
        (
            column
            for column in ["Variable", "variable", "feature"]
            if column in feature_importance.columns
        ),
        None,
    )

    importance_col = next(
        (
            column
            for column in ["Importancia", "importance", "valor"]
            if column in feature_importance.columns
        ),
        None,
    )

    if variable_col is None or importance_col is None:
        raise ValueError(
            "No se encontraron las columnas de variable e importancia."
        )

    top = (
        feature_importance
        .nlargest(top_n, importance_col)
        .sort_values(importance_col)
    )

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.barh(
        top[variable_col],
        top[importance_col],
    )

    ax.set_title(f"{top_n} características más importantes")
    ax.set_xlabel("Importancia")
    ax.set_ylabel("Característica")

    fig.tight_layout()
    return fig


def plot_real_vs_predicted(predictions: pd.DataFrame):
    """Compara la popularidad real contra la predicha"""
    real = predictions["popularidad_real"]
    predicted = predictions["popularidad_predicha"]

    lower = min(real.min(), predicted.min())
    upper = max(real.max(), predicted.max())

    fig, ax = plt.subplots(figsize=(7, 6))

    ax.scatter(
        real,
        predicted,
        alpha=0.2,
        s=12,
    )

    ax.plot(
        [lower, upper],
        [lower, upper],
        linestyle="--",
        linewidth=2,
        color="crimson"
    )

    ax.set_title("Popularidad real vs. popularidad predicha")
    ax.set_xlabel("Popularidad real")
    ax.set_ylabel("Popularidad predicha")

    fig.tight_layout()
    return fig


def plot_residuals(predictions: pd.DataFrame):
    """Genera la distribución de los residuos del modelo"""
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(
        predictions["residuo"].dropna(),
        bins=30,
        edgecolor="black",
    )

    ax.axvline(0, linestyle="--", linewidth=2)

    ax.set_title("Distribución de los residuos")
    ax.set_xlabel("Residuo")
    ax.set_ylabel("Frecuencia")

    fig.tight_layout()
    return fig


def plot_pca_variance(pca_variance: pd.DataFrame):
    """Muestra la varianza explicada y acumulada del ACP"""
    component_col = next(
        (
            column
            for column in ["Component", "component", "Componente"]
            if column in pca_variance.columns
        ),
        None,
    )

    explained_col = next(
        (
            column
            for column in [
                "Explained variance (%)",
                "explained_variance",
                "varianza_explicada",
            ]
            if column in pca_variance.columns
        ),
        None,
    )

    cumulative_col = next(
        (
            column
            for column in [
                "Cumulative variance (%)",
                "cumulative_variance",
                "varianza_acumulada",
            ]
            if column in pca_variance.columns
        ),
        None,
    )

    if None in [component_col, explained_col, cumulative_col]:
        raise ValueError(
            "No se identificaron las columnas necesarias "
            "en pca_variance."
        )

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.bar(
        pca_variance[component_col],
        pca_variance[explained_col],
        label="Varianza explicada",
    )

    ax.plot(
        pca_variance[component_col],
        pca_variance[cumulative_col],
        marker="o",
        label="Varianza acumulada",
    )

    ax.axhline(
        90,
        linestyle="--",
        linewidth=1.5,
        label="90 %",
    )

    ax.set_title("Varianza explicada por componente")
    ax.set_xlabel("Componente")
    ax.set_ylabel("Varianza explicada (%)")
    ax.legend()

    fig.tight_layout()
    return fig


def plot_pca_plane(
    pca_coordinates: pd.DataFrame,
    component_x: str,
    component_y: str,
    clusters: pd.Series | None = None,
):
    """Genera el plano principal para dos componentes"""
    fig, ax = plt.subplots(figsize=(8, 6))

    if clusters is None:
        ax.scatter(
            pca_coordinates[component_x],
            pca_coordinates[component_y],
            alpha=0.15,
            s=10,
        )
    else:
        for cluster in sorted(pd.unique(clusters)):
            mask = clusters == cluster

            ax.scatter(
                pca_coordinates.loc[mask, component_x],
                pca_coordinates.loc[mask, component_y],
                alpha=0.2,
                s=10,
                label=f"Clúster {cluster}",
            )

        ax.legend()

    ax.axhline(0, linestyle="--", linewidth=1)
    ax.axvline(0, linestyle="--", linewidth=1)

    ax.set_title(f"Plano principal: {component_x} vs. {component_y}")
    ax.set_xlabel(component_x)
    ax.set_ylabel(component_y)

    fig.tight_layout()
    return fig


def plot_correlation_circle(
    pca_correlations: pd.DataFrame,
    component_x: str,
    component_y: str,
):
    """Genera el círculo de correlaciones para dos componentes"""
    data = pca_correlations.copy()

    variable_col = (
        "variable"
        if "variable" in data.columns
        else data.columns[0]
    )

    fig, ax = plt.subplots(figsize=(7, 7))

    circle = plt.Circle(
        (0, 0),
        radius=1,
        fill=False,
    )

    ax.add_artist(circle)

    for _, row in data.iterrows():
        x = row[component_x]
        y = row[component_y]

        ax.arrow(
            0,
            0,
            x * 0.95,
            y * 0.95,
            alpha=0.6,
            head_width=0.03,
            length_includes_head=True,
        )

        ax.text(
            x * 1.05,
            y * 1.05,
            str(row[variable_col]),
            ha="center",
            va="center",
        )

    ax.axhline(0, linestyle="--", linewidth=1)
    ax.axvline(0, linestyle="--", linewidth=1)

    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect("equal")

    ax.set_title(
        f"Círculo de correlaciones: "
        f"{component_x} vs. {component_y}"
    )

    ax.set_xlabel(component_x)
    ax.set_ylabel(component_y)

    fig.tight_layout()
    return fig


def plot_elbow(clustering_metrics: pd.DataFrame):
    """Genera la gráfica del método del codo"""
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        clustering_metrics["k"],
        clustering_metrics["inertia"],
        marker="o",
    )

    ax.set_title("Método del codo")
    ax.set_xlabel("Número de clústeres")
    ax.set_ylabel("Inercia")

    fig.tight_layout()
    return fig


def plot_silhouette(clustering_metrics: pd.DataFrame):
    """Genera la gráfica del coeficiente silhouette"""
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        clustering_metrics["k"],
        clustering_metrics["silhouette"],
        marker="o",
    )

    best_row = clustering_metrics.loc[
        clustering_metrics["silhouette"].idxmax()
    ]

    ax.scatter(
        best_row["k"],
        best_row["silhouette"],
        s=80,
        label=f"Mejor k = {int(best_row['k'])}",
    )

    ax.set_title("Coeficiente silhouette")
    ax.set_xlabel("Número de clústeres")
    ax.set_ylabel("Silhouette")
    ax.legend()

    fig.tight_layout()
    return fig


def plot_cluster_profiles(cluster_profiles: pd.DataFrame):
    """Compara las características promedio de los clusters"""
    data = cluster_profiles.copy()

    cluster_col = (
        "cluster"
        if "cluster" in data.columns
        else data.columns[0]
    )

    data = data.set_index(cluster_col).transpose()

    fig, ax = plt.subplots(figsize=(11, 6))

    data.plot(
        kind="bar",
        ax=ax,
    )

    ax.axhline(0, linestyle="--", linewidth=1)

    ax.set_title("Perfil promedio de los clústeres")
    ax.set_xlabel("Característica")
    ax.set_ylabel("Valor estandarizado")
    ax.tick_params(axis="x", rotation=45)

    fig.tight_layout()
    return fig


def plot_cluster_popularity(clustered_data: pd.DataFrame):
    """Compara la popularidad entre los clusters"""
    clusters = sorted(clustered_data["cluster"].dropna().unique())

    values = [
        clustered_data.loc[
            clustered_data["cluster"] == cluster,
            "popularity",
        ].dropna()
        for cluster in clusters
    ]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.boxplot(
        values,
        labels=[f"Clúster {cluster}" for cluster in clusters],
    )

    ax.set_title("Popularidad por clúster")
    ax.set_xlabel("Clúster")
    ax.set_ylabel("Popularidad")

    fig.tight_layout()
    return fig