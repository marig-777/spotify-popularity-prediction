import numpy as np
import pandas as pd

from shiny import Inputs, Outputs, Session, module, render, ui


from data_loader import clustered_data, clustering_metrics, cluster_profiles, pca_coordinates, pca_correlations, pca_variance


from plots import plot_cluster_profiles, plot_correlation_circle, plot_elbow, plot_pca_plane, plot_pca_variance, plot_silhouette


def _component_choices() -> list[str]:
    """
    Obtiene automáticamente las columnas correspondientes
    a las componentes principales.
    """
    return [
        column
        for column in pca_coordinates.columns
        if str(column).upper().startswith("PC")
    ]


def _best_k() -> int:
    """
    Obtiene el número de clústeres con mayor silhouette.
    """
    best_index = clustering_metrics["silhouette"].idxmax()

    return int(
        clustering_metrics.loc[best_index, "k"]
    )


def _variance_column(column_type: str) -> str:
    """
    Identifica las columnas de varianza explicada
    y acumulada sin depender exactamente del idioma
    """
    candidates = {
        "explained": [
            "Explained variance (%)",
            "explained_variance",
            "varianza_explicada",
        ],
        "cumulative": [
            "Cumulative variance (%)",
            "cumulative_variance",
            "varianza_acumulada",
        ],
    }

    for column in candidates[column_type]:
        if column in pca_variance.columns:
            return column

    raise ValueError(
        f"No se encontró la columna de varianza {column_type}."
    )


def _component_number(component_name: str) -> int:
    """
    Convierte nombres como PC1, PC2, etc. en posiciones
    dentro de la tabla de varianza.
    """
    return int(
        "".join(
            character
            for character in component_name
            if character.isdigit()
        )
    )


def _variance_for_component(component_name: str) -> float:
    """
    Devuelve la varianza explicada por una componente
    """
    component_number = _component_number(component_name)
    explained_column = _variance_column("explained")

    return float(
        pca_variance.iloc[
            component_number - 1
        ][explained_column]
    )


def _cluster_profile_text(cluster: int) -> str:
    data = cluster_profiles.copy()

    if "cluster" not in data.columns:
        data = data.reset_index()

        first_column = data.columns[0]

        if first_column != "cluster":
            data = data.rename(
                columns={first_column: "cluster"}
            )

    cluster_row = data[
        data["cluster"].astype(int) == int(cluster)
    ]

    if cluster_row.empty:
        return f"No se encontró información para el clúster {cluster}."

    values = (
        cluster_row
        .drop(columns="cluster")
        .iloc[0]
        .astype(float)
    )

    highest = (
        values
        .sort_values(ascending=False)
        .head(3)
        .index
        .tolist()
    )

    lowest = (
        values
        .sort_values()
        .head(3)
        .index
        .tolist()
    )

    highest_text = ", ".join(highest)
    lowest_text = ", ".join(lowest)

    return (
        f"El clúster {cluster} presenta valores relativamente "
        f"más altos en {highest_text}, y valores más bajos en "
        f"{lowest_text}."
    )


@module.ui
def unsupervised_ui():
    components = _component_choices()

    if len(components) < 2:
        raise ValueError(
            "Se requieren al menos dos componentes principales."
        )

    return ui.nav_panel(
        "Modelado no supervisado",

        ui.div(
            ui.h1("Modelado no supervisado"),
            ui.p(
                "Explora la estructura de las características "
                "musicales mediante ACP y clustering con K-Medias."
            ),
            class_="hero-section",
        ),

        ui.h2("Análisis de componentes principales"),

        ui.layout_columns(
            ui.value_box(
                "Componentes disponibles",
                len(components),
                fill=False,
            ),

            ui.value_box(
                "Varianza en las primeras dos componentes",
                ui.output_text("first_two_variance"),
                fill=False,
            ),

            ui.value_box(
                "Componentes para alcanzar 90 %",
                ui.output_text("components_for_90"),
                fill=False,
            ),

            col_widths=[4, 4, 4],
            fill=False,
        ),

        ui.card(
            ui.card_header(
                "Varianza explicada por componente"
            ),
            ui.card_body(
                ui.output_plot(
                    "pca_variance_plot",
                    height="480px",
                    fill=False,
                )
            ),
            full_screen=True,
            fill=False,
        ),

        ui.layout_sidebar(
            ui.sidebar(
                ui.h4("Componentes a visualizar"),

                ui.input_select(
                    "component_x",
                    "Eje horizontal",
                    choices=components,
                    selected=components[0],
                ),

                ui.input_select(
                    "component_y",
                    "Eje vertical",
                    choices=components,
                    selected=components[1],
                ),

                ui.hr(),

                ui.p(
                    "Selecciona dos componentes distintas "
                    "para actualizar las visualizaciones."
                ),

                width=280,
            ),

            ui.layout_columns(
                ui.card(
                    ui.card_header(
                        "Plano principal"
                    ),
                    ui.card_body(
                        ui.output_plot(
                            "pca_plane_plot",
                            height="560px",
                            fill=False,
                        )
                    ),
                    full_screen=True,
                    fill=False,
                ),

                ui.card(
                    ui.card_header(
                        "Círculo de correlaciones"
                    ),
                    ui.card_body(
                        ui.output_plot(
                            "correlation_circle_plot",
                            height="560px",
                            fill=False,
                        )
                    ),
                    full_screen=True,
                    fill=False,
                ),

                col_widths=[6, 6],
                fill=False,
            ),

            fillable=False,
            fill=False,
        ),

        ui.card(
            ui.card_header("Interpretación del ACP"),
            ui.card_body(
                ui.output_ui("pca_interpretation")
            ),
            fill=False,
        ),

        ui.h2("Clustering con K-Medias"),

        ui.layout_columns(
            ui.value_box(
                "Número de clústeres seleccionado",
                ui.output_text("selected_k"),
                fill=False,
            ),

            ui.value_box(
                "Canciones analizadas",
                ui.output_text("clustered_song_count"),
                fill=False,
            ),

            ui.value_box(
                "Silhouette máximo",
                ui.output_text("best_silhouette"),
                fill=False,
            ),

            col_widths=[4, 4, 4],
            fill=False,
        ),

        ui.layout_columns(
            ui.card(
                ui.card_header("Método del codo"),
                ui.card_body(
                    ui.output_plot(
                        "elbow_plot",
                        height="430px",
                        fill=False,
                    )
                ),
                full_screen=True,
                fill=False,
            ),

            ui.card(
                ui.card_header(
                    "Coeficiente silhouette"
                ),
                ui.card_body(
                    ui.output_plot(
                        "silhouette_plot",
                        height="430px",
                        fill=False,
                    )
                ),
                full_screen=True,
                fill=False,
            ),

            col_widths=[6, 6],
            fill=False,
        ),

        ui.card(
            ui.card_header(
                "Perfil promedio de los clústeres"
            ),
            ui.card_body(
                ui.output_plot(
                    "cluster_profiles_plot",
                    height="560px",
                    fill=False,
                )
            ),
            full_screen=True,
            fill=False,
        ),
    )

@module.server
def unsupervised_server(
    input: Inputs,
    output: Outputs,
    session: Session,
):
    @render.text
    def first_two_variance():
        components = _component_choices()

        total = (
            _variance_for_component(components[0])
            + _variance_for_component(components[1])
        )

        return f"{total:.2f} %"

    @render.text
    def components_for_90():
        cumulative_column = _variance_column("cumulative")

        rows = pca_variance[
            pca_variance[cumulative_column] >= 90
        ]

        if rows.empty:
            return "No alcanzado"

        return str(rows.index[0] + 1)

    @render.plot
    def pca_variance_plot():
        return plot_pca_variance(
            pca_variance
        )

    @render.plot
    def pca_plane_plot():
        component_x = input.component_x()
        component_y = input.component_y()

        if component_x == component_y:
            ui.notification_show(
                "Selecciona componentes distintas.",
                type="warning",
                duration=3,
            )

        clusters = (
            clustered_data["cluster"]
            .reset_index(drop=True)
        )

        coordinates = (
            pca_coordinates
            .reset_index(drop=True)
        )

        return plot_pca_plane(
            coordinates,
            component_x,
            component_y,
            clusters=clusters,
        )

    @render.plot
    def correlation_circle_plot():
        return plot_correlation_circle(
            pca_correlations,
            input.component_x(),
            input.component_y(),
        )

    @render.ui
    def pca_interpretation():
        component_x = input.component_x()
        component_y = input.component_y()

        variance_x = _variance_for_component(
            component_x
        )

        variance_y = _variance_for_component(
            component_y
        )

        total = variance_x + variance_y

        return ui.div(
            ui.p(
                f"{component_x} explica el "
                f"{variance_x:.2f} % de la variabilidad, "
                f"mientras que {component_y} explica el "
                f"{variance_y:.2f} %."
            ),
            ui.p(
                f"En conjunto, ambas componentes representan "
                f"el {total:.2f} % de la variabilidad total. "
                "Esta representación permite visualizar "
                "patrones generales, aunque no contiene toda "
                "la información de las variables originales."
            ),
        )

    @render.text
    def selected_k():
        return str(_best_k())

    @render.text
    def clustered_song_count():
        return f"{len(clustered_data):,}"

    @render.text
    def best_silhouette():
        best_value = (
            clustering_metrics["silhouette"]
            .max()
        )

        return f"{best_value:.3f}"

    @render.plot
    def elbow_plot():
        return plot_elbow(
            clustering_metrics
        )

    @render.plot
    def silhouette_plot():
        return plot_silhouette(
            clustering_metrics
        )

    @render.plot
    def cluster_profiles_plot():
        return plot_cluster_profiles(
            cluster_profiles
        )
    