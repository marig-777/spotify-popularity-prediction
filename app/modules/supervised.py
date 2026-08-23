from shiny import Inputs, Outputs, Session, module, render, ui

from data_loader import (
    feature_importance,
    final_metrics,
    model_results,
    predictions,
)

from plots import (
    plot_feature_importance,
    plot_model_comparison,
    plot_real_vs_predicted,
    plot_residuals,
)


def _get_metric(metric_name: str) -> float:
    """
    Obtiene una métrica final sin depender de mayúsculas
    o minúsculas en su nombre.
    """
    metric_column = next(
        column
        for column in ["metrica", "metric", "Metrica"]
        if column in final_metrics.columns
    )

    value_column = next(
        column
        for column in ["valor", "value", "Valor"]
        if column in final_metrics.columns
    )

    match = final_metrics[
        final_metrics[metric_column]
        .astype(str)
        .str.upper()
        .eq(metric_name.upper())
    ]

    if match.empty:
        return float("nan")

    return float(match.iloc[0][value_column])


def _selected_model_name() -> str:
    """
    Identifica automáticamente el modelo con mayor R² promedio.
    """
    model_column = next(
        column
        for column in ["modelo", "model", "Modelo"]
        if column in model_results.columns
    )

    r2_column = next(
        column
        for column in ["r2_mean", "R2_mean", "R² promedio"]
        if column in model_results.columns
    )

    best_index = model_results[r2_column].idxmax()

    return str(model_results.loc[best_index, model_column])


@module.ui
def supervised_ui():
    return ui.nav_panel(
        "Modelo predictivo",

        ui.div(
            ui.h1("Modelo predictivo"),
            ui.p(
                "Consulta la comparación de los modelos de regresión "
                "y el desempeño del modelo seleccionado."
            ),
            class_="hero-section",
        ),

        ui.card(
            ui.card_header("Modelo seleccionado"),
            ui.div(
                ui.div(
                    ui.h2(ui.output_text("selected_model")),
                    ui.p(
                        "Modelo con mejor desempeño durante "
                        "la validación cruzada."
                    ),
                ),
                class_="selected-model",
            ),
            fill=False,
        ),

        ui.layout_columns(
            ui.value_box(
                "R²",
                ui.output_text("r2_value"),
                fill=False,
            ),

            ui.value_box(
                "MAE",
                ui.output_text("mae_value"),
                fill=False,
            ),

            ui.value_box(
                "RMSE",
                ui.output_text("rmse_value"),
                fill=False,
            ),

            col_widths=[4, 4, 4],
            fill=False,
        ),

        ui.card(
            ui.card_header("Comparación entre modelos"),
            ui.card_body(
                ui.output_plot(
                    "model_comparison",
                    height="430px",
                    fill=False,
                )
            ),
            full_screen=True,
            fill=False,
        ),

        ui.layout_columns(
            ui.card(
                ui.card_header("Importancia de las características"),
                ui.card_body(
                    ui.output_plot(
                        "feature_importance_plot",
                        height="520px",
                        fill=False,
                    )
                ),
                full_screen=True,
                fill=False,
            ),

            ui.card(
                ui.card_header("Popularidad real vs. predicha"),
                ui.card_body(
                    ui.output_plot(
                        "real_vs_predicted",
                        height="520px",
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
            ui.card_header("Distribución de los errores"),
            ui.card_body(
                ui.output_plot(
                    "residual_distribution",
                    height="430px",
                    fill=False,
                )
            ),
            full_screen=True,
            fill=False,
        ),

        ui.card(
            ui.card_header("Interpretación general"),
            ui.output_ui("model_interpretation"),
            fill=False,
        ),
    )


@module.server
def supervised_server(
    input: Inputs,
    output: Outputs,
    session: Session,
):
    @render.text
    def selected_model():
        return _selected_model_name()

    @render.text
    def r2_value():
        return f"{_get_metric('R2'):.3f}"

    @render.text
    def mae_value():
        return f"{_get_metric('MAE'):.2f}"

    @render.text
    def rmse_value():
        return f"{_get_metric('RMSE'):.2f}"

    @render.plot
    def model_comparison():
        return plot_model_comparison(model_results)

    @render.plot
    def feature_importance_plot():
        return plot_feature_importance(
            feature_importance,
            top_n=15,
        )

    @render.plot
    def real_vs_predicted():
        return plot_real_vs_predicted(predictions)

    @render.plot
    def residual_distribution():
        return plot_residuals(predictions)

    @render.ui
    def model_interpretation():
        model_name = _selected_model_name()
        r2 = _get_metric("R2")

        return ui.div(
            ui.p(
                f"El modelo {model_name} obtuvo el mejor desempeño "
                "durante la comparación realizada mediante validación cruzada."
            ),
            ui.p(
                f"En el conjunto de prueba alcanzó un R² de {r2:.2f}, "
                f"por lo que explica aproximadamente el {r2 * 100:.0f} % "
                "de la variabilidad observada en la popularidad."
            ),
        )