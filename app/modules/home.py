from shiny import ui

from data_loader import df, clustered_data


def home_ui():
    return ui.nav_panel(
        "Inicio",

        ui.div(
            ui.h1("Análisis y predicción de la popularidad en Spotify"),
            ui.p(
                "Dashboard interactivo para explorar los datos, "
                "los resultados del modelado y la predicción de popularidad."
            ),
            class_="hero-section",
        ),

        ui.layout_columns(
            ui.value_box(
                "Canciones",
                f"{len(df):,}",
            ),
            ui.value_box(
                "Variables",
                df.shape[1],
            ),
            ui.value_box(
                "Clústeres",
                clustered_data["cluster"].nunique(),
            ),
            ui.value_box(
                "Popularidad promedio",
                f"{df['popularity'].mean():.2f}",
            ),
            col_widths=[3, 3, 3, 3],
        ),

        ui.layout_columns(
            ui.card(
                ui.card_header("Acerca del proyecto"),
                ui.p(
                    "Este proyecto analiza un conjunto de canciones de Spotify "
                    "con el objetivo de estudiar la relación entre sus "
                    "características musicales y su nivel de popularidad."
                ),
                ui.p(
                    "El dashboard integra los resultados del análisis "
                    "exploratorio, la comparación de modelos de regresión, "
                    "el análisis de componentes principales y el clustering "
                    "mediante K-Medias."
                ),
            ),

            col_widths=[12],
        ),
    )