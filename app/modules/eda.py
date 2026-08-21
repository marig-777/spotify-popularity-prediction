from shiny import Inputs, Outputs, Session, module, reactive, render, ui

from data_loader import df
from plots import (
    plot_boxplot,
    plot_correlation_heatmap,
    plot_histogram,
)


def _numeric_variables():
    """
    Selecciona variables numéricas con suficiente variación
    para analizarlas como continuas
    """
    return [
        column
        for column in df.select_dtypes(include="number").columns
        if df[column].nunique(dropna=True) > 10
    ]


def _genre_choices():
    """
    Genera las opciones del selector directamente desde el dataset
    """
    genres = sorted(
        df["track_genre"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    return ["Todos"] + genres


@module.ui
def eda_ui():
    numeric_variables = _numeric_variables()

    return ui.nav_panel(
        "Exploración",

        ui.div(
            ui.h1("Exploración de datos"),
            ui.p(
                "Explora la distribución de las características musicales "
                "y filtra las canciones por género."
            ),
            class_="hero-section",
        ),

        ui.layout_sidebar(
            ui.sidebar(
                ui.input_selectize(
                    "genre",
                    "Género musical",
                    choices=_genre_choices(),
                    selected="Todos",
                ),

                ui.input_select(
                    "variable",
                    "Variable numérica",
                    choices=numeric_variables,
                    selected=numeric_variables[0],
                ),

                ui.hr(),

                ui.p(
                    "Las gráficas y estadísticas se actualizan "
                    "automáticamente con los filtros seleccionados."
                ),
            ),

            ui.layout_columns(
                ui.value_box(
                    "Canciones filtradas",
                    ui.output_text("song_count")
                ),

                ui.value_box(
                    "Media",
                    ui.output_text("mean_value")
                ),

                ui.value_box(
                    "Mediana",
                    ui.output_text("median_value")
                ),

                ui.value_box(
                    "Desviación estándar",
                    ui.output_text("std_value")
                ),

                col_widths=[3, 3, 3, 3],
            ),

            ui.layout_columns(
                ui.card(
                    ui.card_header("Distribución"),
                    ui.output_plot("histogram", height="420px"),
                ),

                ui.card(
                    ui.card_header("Boxplot"),
                    ui.output_plot("boxplot", height="420px"),
                ),

                col_widths=[6, 6],
            ),

            ui.card(
                ui.card_header("Resumen descriptivo"),
                ui.output_data_frame("summary_table"),
            ),

            ui.card(
                ui.card_header("Matriz de correlaciones"),
                ui.output_plot("correlation_heatmap", height="650px"),
            )
        ),
    )


@module.server
def eda_server(
    input: Inputs,
    output: Outputs,
    session: Session,
):
    @reactive.calc
    def filtered_data():
        data = df.copy()

        if input.genre() != "Todos":
            data = data[
                data["track_genre"].astype(str) == input.genre()
            ]

        return data

    @render.text
    def song_count():
        return f"{len(filtered_data()):,}"

    @render.text
    def mean_value():
        variable = input.variable()
        value = filtered_data()[variable].mean()

        return f"{value:.2f}"

    @render.text
    def median_value():
        variable = input.variable()
        value = filtered_data()[variable].median()

        return f"{value:.2f}"

    @render.text
    def std_value():
        variable = input.variable()
        value = filtered_data()[variable].std()

        return f"{value:.2f}"

    @render.plot
    def histogram():
        return plot_histogram(
            filtered_data(),
            input.variable(),
        )

    @render.plot
    def boxplot():
        return plot_boxplot(
            filtered_data(),
            input.variable(),
        )

    @render.plot
    def correlation_heatmap():
        return plot_correlation_heatmap(
            filtered_data()
        )

    @render.data_frame
    def summary_table():
        variable = input.variable()

        summary = (
            filtered_data()[variable]
            .describe()
            .rename_axis("estadístico")
            .reset_index(name="valor")
        )

        summary["valor"] = summary["valor"].round(3)

        return render.DataGrid(
            summary,
            width="100%",
            filters=False,
        )

    @render.data_frame
    def filtered_table():
        columns_to_show = [
            column
            for column in [
                "track_name",
                "artists",
                "track_genre",
                "popularity",
                input.variable(),
            ]
            if column in filtered_data().columns
        ]

        table = (
            filtered_data()[columns_to_show]
            .head(1000)
            .reset_index(drop=True)
        )

        return render.DataGrid(
            table,
            width="100%",
            height="420px",
            filters=True,
        )