"""
Edward's original app
"""

from collections import deque

import numpy as np
import panel as pn
from bokeh import models
from bokeh import plotting
from bokeh.palettes import Blues256
from bokeh.palettes import Oranges256
from bokeh.palettes import Reds256
from bokeh.palettes import diverging_palette

from laser_model.england_wales.model import EnglandWalesModel
from laser_model.england_wales.params import get_parameters
from laser_model.england_wales.scenario import get_scenario
from laser_model.mixing import init_gravity_diffusion

PRIMARY_COLOR = "#0072B5"
SECONDARY_COLOR = "#B54300"

pn.extension(design="material", sizing_mode="stretch_width")


params = get_parameters({})
scenario = get_scenario()
model = EnglandWalesModel(parameters=params, scenario=scenario)
model.run()

# reset functions

# memoize the function (https://panel.holoviz.org/how_to/caching/memoization.html)
@pn.cache
def get_data():
    return get_scenario()


def reset_parameters():
    return get_parameters({})


def reset_state():
    model.init_state(get_data(), reset_parameters())


reset_state()


def step():
    if not hasattr(step, "tick"):
        step.tick = 0
    model.step(step.tick)
    step.tick += 1


beta_slider = pn.widgets.FloatSlider(value=params.beta, start=0, end=50, step=1, name="beta")


def on_beta_change(value):
    params.beta = value


bound_beta = pn.bind(on_beta_change, value=beta_slider)

seasonality_slider = pn.widgets.FloatSlider(value=params.seasonality, start=0, end=0.3, step=0.02, name="seasonality")


def on_seasonality_change(value):
    params.seasonality = value


bound_seasonality = pn.bind(on_seasonality_change, value=seasonality_slider)

demog_scale_slider = pn.widgets.FloatSlider(value=params.demog_scale, start=0.1, end=1.5, step=0.05, name="demog_scale")


def on_demog_scale_change(value):
    params.demog_scale = value
    params.biweek_avg_births = params.demog_scale * params.births / 26.0
    params.biweek_death_prob = params.demog_scale * params.births / params.population / 26.0


bound_demog_scale = pn.bind(on_demog_scale_change, value=demog_scale_slider)


mixing_scale_slider = pn.widgets.FloatSlider(value=np.log10(params.mixing_scale), start=-4, end=-2, name="log10(mixing_scale)")


def on_mixing_scale_change(value):
    params.mixing_scale = np.power(10, value)
    params.mixing = init_gravity_diffusion(scenario, params.mixing_scale, params.distance_exponent)


bound_mixing_scale = pn.bind(on_mixing_scale_change, value=mixing_scale_slider)

distance_exponent_slider = pn.widgets.FloatSlider(value=params.distance_exponent, start=0.5, end=2.5, step=0.1, name="distance_exponent")


def on_distance_exponent_change(value):
    params.distance_exponent = value
    params.mixing = init_gravity_diffusion(scenario, params.mixing_scale, params.distance_exponent)


bound_distance_exponent = pn.bind(on_distance_exponent_change, value=distance_exponent_slider)

source = models.ColumnDataSource( {
        "name": scenario.index,
        "x": scenario.Long,
        "y": scenario.Lat,
        "size": 0.03 * np.sqrt(scenario.population),
        "population": scenario.population,
        "births": scenario.births,
        "prevalence": model.nodes.states[1] / model.nodes.states.sum(axis=0),
        "reff": params.beta * model.nodes.states[0] / model.nodes.states.sum(axis=0)
    } )

hover = models.HoverTool(
    tooltips=[
        ("name", "@name"),
        ("population", "@population"),
        ("births", "@births"),
    ]
)

prev_cmap = models.LogColorMapper(palette=Reds256[::-1], low=1e-4, high=0.01)
reff_cmap = models.LogColorMapper(palette=diverging_palette(Blues256, Oranges256, n=256), low=0.25, high=4.0)

prev_scatter = plotting.figure(
    x_axis_label="Longitude",
    y_axis_label="Latitude",
    title="Prevalence",
    width=500,
    height=500,
)
prev_scatter.add_tools(hover)
prev_scatter.scatter(x="x", y="y", size="size", color={"field": "prevalence", "transform": prev_cmap}, source=source, alpha=0.5)

reff_scatter = plotting.figure(
    x_axis_label="Longitude",
    y_axis_label="Latitude",
    title="Effective reproductive number",
    width=500,
    height=500,
)
reff_scatter.add_tools(hover)
reff_scatter.scatter(x="x", y="y", size="size", color={"field": "reff", "transform": reff_cmap}, source=source, alpha=0.5)


ts_source = models.ColumnDataSource({"time": np.arange(0, 10 * 26), "prevalence": np.zeros(10 * 26)})

prev_ts = plotting.figure(x_axis_label="Time (years)", y_axis_label="Prevalence (%)", width=500, height=200)
prev_ts.line(x="time", y="prevalence", source=ts_source, color="red")


time_ts_list = deque()
prev_ts_list = deque()


def stream():
    step()

    prev_scatter.title.text = f"Prevalence (year = {step.tick/26.:.2f})"
    source.data["prevalence"] = model.nodes.states[1] / model.nodes.states.sum(axis=0)
    source.data["reff"] = params.beta * model.nodes.states[0] / model.nodes.states.sum(axis=0)

    time_ts_list.append(step.tick / 26.0)
    prev_ts_list.append(100 * model.nodes.states[1].sum() / model.nodes.states.sum())  # (prev in %)
    # prev_ts_list.append(np.random.poisson(lam=state[:, 1].sum()/2000.))  # (downsampled case counts)
    # prev_ts_list.append(100 * (state[:, 1] > 0).sum() / len(state[:, 1]))  # (non-zero prevalence %)
    if len(time_ts_list) > 10 * 26:
        time_ts_list.popleft()
        prev_ts_list.popleft()
    ts_source.data = {"time": list(time_ts_list), "prevalence": list(prev_ts_list)}


callback_period = 50
callback = pn.state.add_periodic_callback(stream, callback_period)

speed_slider = pn.widgets.FloatSlider(value=callback_period, start=10, end=200, step=10, name="refresh rate (ms)")


def on_speed_change(value):
    callback.period = value


bound_speed = pn.bind(on_speed_change, value=speed_slider)

reset_button = pn.widgets.Button(name="Reset", button_type="primary")


def reset(event):
    params = reset_parameters()
    reset_state()
    step.tick = 0
    beta_slider.value = params.beta
    seasonality_slider.value = params.seasonality
    demog_scale_slider.value = params.demog_scale
    mixing_scale_slider.value = np.log10(params.mixing_scale)
    distance_exponent_slider.value = params.distance_exponent
    speed_slider.value = callback_period
    if not callback.running:
        callback.start()
    time_ts_list.clear()
    prev_ts_list.clear()


reset_button.on_click(reset)

pause_button = pn.widgets.Toggle(name="Pause/Resume", value=True)
pause_button.link(callback, bidirectional=True, value="running")

sliders = pn.Column(
    "### Simulation parameters",
    pn.Row(beta_slider, bound_beta),
    pn.Row(seasonality_slider, bound_seasonality),
    pn.Row(demog_scale_slider, bound_demog_scale),
    pn.layout.Divider(),
    "### Mixing parameters",
    pn.Row(mixing_scale_slider, bound_mixing_scale),
    pn.Row(distance_exponent_slider, bound_distance_exponent),
    pn.layout.Divider(),
    "### Playback controls",
    pn.Row(speed_slider, bound_speed),
    pn.Row(reset_button, pause_button),
)

template = pn.template.MaterialTemplate(
    site="numpy demo",
    title="Interactive Spatial Simulation",
    sidebar=[sliders],
    main=[pn.Row(prev_scatter, reff_scatter), prev_ts],
)

pn.serve(template)
