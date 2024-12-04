"""
docker run -it -p 4243:4243 linusseelinger/benchmark-muq-beam-propagation:latest

https://github.com/UM-Bridge/benchmarks/tree/main/benchmarks/muq-beam-propagation
https://um-bridge-benchmarks.readthedocs.io/en/docs/forward-benchmarks/muq-beam-propagation.html
"""
# https://github.com/InstituteforDiseaseModeling/laser-cohorts/blob/main/bokeh/england_wales_app.py
import argparse
import umbridge
import numpy as np
import panel as pn
import sciris as sc
from bokeh import models
from bokeh import plotting
import viz_umbridge as vu

PRIMARY_COLOR = "#780078" # UM-Bridge purple
SECONDARY_COLOR = "#F5A91E" # UM-Bridge yellow

# create a fixed size float buffer with additional histogram functionality
class QFixedSizeBuffer(vu.FixedSizeFloatBuffer):
    def __init__(self, n, hist_bins=25):
        super().__init__(n)

        # initialize the bins
        self.hist = None
        self.hist_bin_centers = None
        if isinstance(hist_bins, (list, np.ndarray)):
            self.num_hist_bins = len(hist_bins)
            self.hist_bins = hist_bins
        else:
            self.num_hist_bins = hist_bins
            self.hist_bins = None        
        self.init_hist()

    # intialize the histogram
    def init_hist(self):
        if np.all(np.isnan(self.buffer)):
            # self.hist, self.hist_bins = np.histogram(np.zeros(self.n), bins=self.num_hist_bins)
            self.hist = [0,0,0]
            self.hist_bins = [0,1,2,4]
        else:
            self.hist, self.hist_bins = np.histogram(self.buffer[np.isfinite(self.buffer)], bins=self.num_hist_bins)
        # remove the last bin edge for bokeh's step plot
        self.hist_bin_centers = [0.5*(self.hist_bins[i] + self.hist_bins[i+1]) for i in range(len(self.hist_bins)-1)]

    # update the histogram
    def update_hist(self):
        self.hist, _ = np.histogram(self.buffer, bins=self.hist_bins)


# initialize umbridge connection
# Read URL from command line argument
parser = argparse.ArgumentParser(description='Minimal HTTP model demo.')
parser.add_argument('--url', metavar='url', type=str, default='http://localhost:4243',
                    help='the URL at which the model is running, for example http://localhost:4243 (default: http://localhost:4243)')
args = parser.parse_args()
print(f"Connecting to host URL {args.url}")
# Print models supported by server
print(umbridge.supported_models(args.url))

# initialize parameters for the prior
def reset_params():
    return sc.objdict({
    "m1": 1.025, 
    "m2": 1.025,
    "m3": 1.025,
    "width": 0.02,
}
)
prior_params = reset_params()

# Set up a model by connecting to URL and selecting the "forward" model
model = umbridge.HTTPModel(args.url, "forward")
num_beam_elements = model.get_output_sizes()[0]
print(f"Number of beam elements: {num_beam_elements}")


# initialize results we are tracking / plotting
buffer_size = 100
beam_values_buffer = vu.FixedSizeObjectBuffer(buffer_size, placeholder=num_beam_elements*[0])
Q1_buffer = QFixedSizeBuffer(buffer_size)
Q2_buffer = QFixedSizeBuffer(buffer_size)

beam_source = models.ColumnDataSource({
    "beam_indices": [np.arange(num_beam_elements) for _ in range(beam_values_buffer.n)],
    "beam_values": beam_values_buffer.buffer,
    "Q1_element": Q1_buffer.n*[9],
    "Q1_buffer": Q1_buffer.buffer,
    "Q2_element": Q2_buffer.n*[24],
    "Q2_buffer": Q2_buffer.buffer,
})
Q_source = models.ColumnDataSource({
    "Q1_hist": Q1_buffer.hist,
    "Q1_hist_bins": Q1_buffer.hist_bins,
    "Q2_hist": Q2_buffer.hist,
    "Q2_hist_bins": Q2_buffer.hist_bins,
})

# create step function for evaluating the forward model
def step():

    # draw 3 RV from gaussian distribution
    param = list(np.maximum(0, prior_params.width*np.random.randn(3) + np.array([prior_params.m1, prior_params.m2, prior_params.m3])))
    beam_values_buffer.add(model([param])[0])

    # update the values of interest
    Q1_buffer.add(beam_values_buffer.buffer[Q1_buffer.get_index()][9])
    Q2_buffer.add(beam_values_buffer.buffer[Q2_buffer.get_index()][24])

    # update the histograms
    Q1_buffer.init_hist()
    Q2_buffer.init_hist()

    # update sources
    beam_source.data["beam_values"] = beam_values_buffer.buffer
    beam_source.data["Q1_buffer"] = Q1_buffer.buffer
    beam_source.data["Q2_buffer"] = Q2_buffer.buffer
    Q_source.data["Q1_hist"] = Q1_buffer.hist
    Q_source.data["Q1_hist_bins"] = Q1_buffer.hist_bin_centers
    Q_source.data["Q2_hist"] = Q2_buffer.hist
    Q_source.data["Q2_hist_bins"] = Q2_buffer.hist_bin_centers


# create streaming function for updating the plots
def stream():
    if not hasattr(stream, 'n'):
        stream.n = 0
    stream.n += 1
    step()

    beam_plot.title.text = f"N={stream.n}"

# initialize panel
pn.extension(design="material", sizing_mode="stretch_width")


# initialize sliders
slider_m1 = pn.widgets.FloatSlider(value=prior_params.m1, start=0, end=2, step=.05, name="M1")
def on_m1_change(value):
    prior_params.m1 = value
bound_m1 = pn.bind(on_m1_change, value=slider_m1)

slider_m2 = pn.widgets.FloatSlider(value=prior_params.m2, start=0, end=2, step=.05, name="M2")
def on_m2_change(value):
    prior_params.m2 = value
bound_m2 = pn.bind(on_m2_change, value=slider_m1)

slider_m3 = pn.widgets.FloatSlider(value=prior_params.m3, start=0, end=2, step=.05, name="M3")
def on_m3_change(value):
    prior_params.m3 = value
bound_m3 = pn.bind(on_m3_change, value=slider_m3)

slider_width = pn.widgets.FloatSlider(value=prior_params.width, start=0, end=1, step=.01, name="Width")
def on_width_change(value):
    prior_params.width = value
bound_width = pn.bind(on_width_change, value=slider_width)

callback = pn.state.add_periodic_callback(stream, (callback_period := 50), start=False)
slider_speed = pn.widgets.FloatSlider(value=callback_period, start=1, end=500, step=10, name="refresh rate (ms)")
def on_speed_change(value):
    callback.period = value
bound_speed = pn.bind(on_speed_change, value=slider_speed)

def reset(event):
    prior_params = reset_params()
    slider_m1.value = prior_params.m1
    slider_m2.value = prior_params.m2
    slider_m3.value = prior_params.m3
    slider_width.value = prior_params.width
    slider_speed.value = callback_period
    stream.n = 0
    if not callback.running:
        callback.start()
reset_button = pn.widgets.Button(name="Reset")
reset_button.on_click(reset)

pause_button = pn.widgets.Toggle(name="Start/Stop", value=False, button_type="primary")
pause_button.link(callback, bidirectional=True, value="running")

# configure subplots
beam_plot = plotting.figure(
    x_axis_label = "Beam Element",
    y_axis_label = "Value",
    title = "N=0",
    y_range = [0, 1200],
    width = 500,
    height = 400,
)
beam_plot.multi_line(xs="beam_indices", ys="beam_values", 
                     line_color='blue', line_width=0.5, alpha=0.2, source=beam_source)
beam_plot.scatter(x="Q1_element", y="Q1_buffer", color='red', marker='x', source=beam_source)
beam_plot.scatter(x="Q2_element", y="Q2_buffer", color='red', marker='x', source=beam_source)

Q1_plot = plotting.figure(
    x_axis_label = "Q1",
    title = "Q1",
    width = 300,
    height = 400,
    x_range = [0, 1200]
)
Q1_plot.step(x="Q1_hist_bins", y="Q1_hist", line_color='blue', mode="center", line_width=2, source=Q_source)

Q2_plot = plotting.figure(
    x_axis_label = "Q2",
    title = "Q2",
    width = 300,
    height = 400,
    x_range = [0, 1200]
)
Q2_plot.step(x="Q2_hist_bins", y="Q2_hist", line_color='blue', mode="center", line_width=2, source=Q_source)

sliders = pn.Column(
    "### Prior parameters",
    pn.Row(slider_m1, bound_m1),
    pn.Row(slider_m2, bound_m2),
    pn.Row(slider_m3, bound_m3),
    pn.Row(slider_width, bound_width),
    pn.layout.Divider(),
    "### Playback controls",
    pn.Row(slider_speed, bound_speed),
    pn.Row(reset_button, pause_button),
)

template = pn.template.MaterialTemplate(
    site="MUQ Beam Forward Simulation",
    title="MUQ Beam Forward Simulation",
    header_background=PRIMARY_COLOR,
    sidebar=[sliders],
    main=[pn.Row(beam_plot, Q1_plot, Q2_plot)],
)

pn.serve(template)

print("done")