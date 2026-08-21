import numpy as np
import pandas as pd

from shiny import (
    Inputs,
    Outputs,
    Session,
    module,
    reactive,
    render,
    ui,
)

from data_loader import df, model

preprocessor = model.named_steps["preprocessor"]


def _get_transformer_columns(transformer_name: str) -> list[str]:
    """
    Obtiene las columnas utilizadas por un transformador
    del ColumnTransformer.
    """
    for name, transformer, columns in preprocessor.transformers_:
        if name == transformer_name:
            return list(columns)

    raise ValueError(
        f"No se encontró el transformador '{transformer_name}'."
    )


NUMERICAL_FEATURES = _get_transformer_columns("num")
CATEGORICAL_FEATURES = _get_transformer_columns("cat")


def _get_encoder():
    """
    Obtiene el OneHotEncoder utilizado por el pipeline.
    """
    return preprocessor.named_transformers_["cat"]


ENCODER = _get_encoder()


def _format_label(variable: str) -> str:
    """
    Convierte nombres como duration_ms en Duration ms.
    """
    return variable.replace("_", " ").capitalize()


def _numeric_input(variable: str):
    """
    Crea automáticamente un control numérico usando
    el rango y la mediana observados en el dataset.
    """
    values = df[variable].dropna().astype(float)

    minimum = float(values.min())
    maximum = float(values.max())
    median = float(values.median())

    variable_range = maximum - minimum

    if variable_range <= 2:
        step = 0.01
    elif variable_range <= 100:
        step = 0.1
    else:
        step = max(1, round(variable_range / 100))


    if variable_range > 1000:
        return ui.input_numeric(
            variable,
            _format_label(variable),
            value=round(median, 2),
            min=minimum,
            max=maximum,
            step=step,
        )

    return ui.input_slider(
        variable,
        _format_label(variable),
        min=minimum,
        max=maximum,
        value=median,
        step=step,
    )


def _categorical_choices(variable: str) -> dict[str, str]:
    """
    Obtiene las categorías aprendidas por el encoder.
    """
    position = CATEGORICAL_FEATURES.index(variable)
    categories = ENCODER.categories_[position]

    return {
        str(category): str(category)
        for category in categories
    }


def _default_category(variable: str) -> str:
    """
    Usa como valor inicial la categoría más frecuente
    del dataset
    """
    mode = df[variable].mode(dropna=True)

    if not mode.empty:
        return str(mode.iloc[0])

    return next(iter(_categorical_choices(variable)))


def _recover_category_value(variable: str, selected_value: str):
    """
    Convierte el texto seleccionado al tipo original
    que conoció el OneHotEncoder
    """
    position = CATEGORICAL_FEATURES.index(variable)
    categories = ENCODER.categories_[position]

    for category in categories:
        if str(category) == str(selected_value):
            return category

    raise ValueError(
        f"El valor '{selected_value}' no pertenece a {variable}."
    )


@module.ui
def prediction_ui():
    numerical_inputs = [
        _numeric_input(variable)
        for variable in NUMERICAL_FEATURES
    ]

    categorical_inputs = [
        ui.input_select(
            variable,
            _format_label(variable),
            choices=_categorical_choices(variable),
            selected=_default_category(variable),
        )
        for variable in CATEGORICAL_FEATURES
    ]

    return ui.nav_panel(
        "Explora una canción",

        ui.div(
            ui.h1("Explora una canción"),
            ui.p(
                "Modifica las características musicales para obtener "
                "una estimación de su popularidad."
            ),
            class_="hero-section",
        ),

        ui.layout_columns(
            ui.card(
                ui.card_header("Características musicales"),

                ui.card_body(
                    ui.layout_columns(
                        *numerical_inputs,
                        col_widths=[6, 6],
                        fill=False,
                    )
                ),

                fill=False,
            ),

            ui.card(
                ui.card_header("Características generales"),

                ui.card_body(
                    *categorical_inputs,

                    ui.hr(),

                    ui.input_action_button(
                        "predict_button",
                        "Estimar popularidad",
                        class_="btn-predict",
                    ),
                ),

                fill=False,
            ),

            col_widths=[8, 4],
            fill=False,
        ),

        ui.card(
            ui.card_body(
                ui.div(
                    ui.span("Popularidad estimada"),
                    ui.output_text(
                        "predicted_popularity"
                    ),
                    ui.p(
                        "Valor estimado por el modelo Random Forest "
                        "en una escala de 0 a 100."
                    ),
                    class_="prediction-result",
                )
            ),
            fill=False,
        ),

        ui.card(
            ui.card_header("Características ingresadas"),
            ui.card_body(
                ui.output_data_frame(
                    "prediction_input_table"
                )
            ),
            fill=False,
        ),

        ui.card(
            ui.card_header("Consideraciones"),
            ui.card_body(
                ui.p(
                    "La estimación se basa únicamente en las "
                    "características utilizadas durante el entrenamiento. "
                    "Otros factores, como la promoción, el artista, las "
                    "tendencias y el momento de lanzamiento, no están "
                    "incluidos en el modelo."
                )
            ),
            fill=False,
        ),
    )



@module.server
def prediction_server(
    input: Inputs,
    output: Outputs,
    session: Session,
):
    @reactive.calc
    def input_data():
        row = {}

        for variable in NUMERICAL_FEATURES:
            value = getattr(input, variable)()
            row[variable] = float(value)

        for variable in CATEGORICAL_FEATURES:
            selected_value = getattr(input, variable)()

            row[variable] = _recover_category_value(
                variable,
                selected_value,
            )

        return pd.DataFrame([row])

    @reactive.calc
    @reactive.event(input.predict_button)
    def prediction_value():
        prediction = float(
            model.predict(input_data())[0]
        )

        # La popularidad del dataset se encuentra entre 0 y 100.
        return float(np.clip(prediction, 0, 100))

    @render.text
    def predicted_popularity():
        if input.predict_button() == 0:
            return "—"

        return f"{prediction_value():.1f} / 100"

    @render.data_frame
    def prediction_input_table():
        table = input_data().copy()

        table.columns = [
            _format_label(column)
            for column in table.columns
        ]

        return render.DataGrid(
            table,
            width="100%",
            filters=False,
        )