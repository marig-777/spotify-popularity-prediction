from pathlib import Path

from shiny import App, ui

from modules.home import home_ui
from modules.eda import eda_ui, eda_server
from modules.supervised import supervised_server, supervised_ui
from modules.unsupervised import unsupervised_server, unsupervised_ui
from modules.prediction import prediction_server, prediction_ui

navbar = ui.page_navbar(
    home_ui(),

    eda_ui("eda"),

    supervised_ui("supervised"),

    unsupervised_ui("unsupervised"),

    prediction_ui("prediction"),

    title="Spotify Popularity Dashboard",
    fillable=False,
)


app_ui = ui.TagList(
    ui.include_css(Path(__file__).parent / "styles.css"),
    navbar,
)


def server(input, output, session):
    eda_server("eda")
    supervised_server("supervised")
    unsupervised_server("unsupervised")
    prediction_server("prediction")


app = App(app_ui, server)