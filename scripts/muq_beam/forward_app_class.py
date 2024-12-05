import argparse
import umbridge
import numpy as np
import panel as pn
import sciris as sc
from bokeh import models
from bokeh import plotting
import viz_umbridge as vu

PRIMARY_COLOR = "#780078"  # UM-Bridge purple
SECONDARY_COLOR = "#F5A91E"  # UM-Bridge yellow

class QFixedSizeBuffer(vu.FixedSizeFloatBuffer):
    def __init__(self, n, hist_bins=25):
        super().__init__(n)
        self.hist = None
        self.hist_bin_centers = None
        if isinstance(hist_bins, (list, np.ndarray)):
            self.num_hist_bins = len(hist_bins)
            self.hist_bins = hist_bins
        else:
            self.num_hist_bins = hist_bins
            self.hist_bins = None        
        self.init_hist()

    def init_hist(self):
        if np.all(np.isnan(self.buffer)):
            self.hist = [0, 0, 0]
            self.hist_bins = [0, 1, 2, 4]
        else:
            self.hist, self.hist_bins = np.histogram(
                self.buffer[np.isfinite(self.buffer)], bins=self.num_hist_bins
            )
        self.hist_bin_centers = [
            0.5 * (self.hist_bins[i] + self.hist_bins[i + 1]) for i in range(len(self.hist_bins) - 1)
        ]

    def update_hist(self):
        self.hist, _ = np.histogram(self.buffer, bins=self.hist_bins)

class UmbridgePanelApp:
    def __init__(self, url,  model_name="forward"):
        self.url = url
        self.model_name = model_name
        self.callback_period = 50
        self.prior_params = self.reset_params()
        self.connect_model()
        self.initialize_buffers()
        self.initialize_data_sources()
        self.initialize_widgets()
        self.setup_plots()
        self.setup_template()

    def reset_params(self):
        return sc.objdict({
            "m1": 1.025,
            "m2": 1.025,
            "m3": 1.025,
            "width": 0.02,
        })

    def connect_model(self):
        print(f"Connecting to host URL {self.url}")
        print(umbridge.supported_models(self.url))
        self.model = umbridge.HTTPModel(self.url, self.model_name)
        self.num_beam_elements = self.model.get_output_sizes()[0]
        print(f"Number of beam elements: {self.num_beam_elements}")

    def initialize_buffers(self):
        buffer_size = 100
        self.beam_values_buffer = vu.FixedSizeObjectBuffer(
            buffer_size, placeholder=self.num_beam_elements * [0]
        )
        self.Q1_buffer = QFixedSizeBuffer(buffer_size)
        self.Q2_buffer = QFixedSizeBuffer(buffer_size)

    def initialize_data_sources(self):
        self.beam_source = models.ColumnDataSource({
            "beam_indices": [np.arange(self.num_beam_elements) for _ in range(self.beam_values_buffer.n)],
            "beam_values": self.beam_values_buffer.buffer,
            "Q1_element": self.Q1_buffer.n * [9],
            "Q1_buffer": self.Q1_buffer.buffer,
            "Q2_element": self.Q2_buffer.n * [24],
            "Q2_buffer": self.Q2_buffer.buffer,
        })
        self.Q_source = models.ColumnDataSource({
            "Q1_hist": self.Q1_buffer.hist,
            "Q1_hist_bins": self.Q1_buffer.hist_bins,
            "Q2_hist": self.Q2_buffer.hist,
            "Q2_hist_bins": self.Q2_buffer.hist_bins,
        })

    def step(self):
        param = list(
            np.maximum(
                0,
                self.prior_params.width * np.random.randn(3) + np.array([
                    self.prior_params.m1,
                    self.prior_params.m2,
                    self.prior_params.m3,
                ]),
            )
        )
        self.beam_values_buffer.add(self.model([param])[0])
        self.Q1_buffer.add(self.beam_values_buffer.buffer[self.Q1_buffer.get_index()][9])
        self.Q2_buffer.add(self.beam_values_buffer.buffer[self.Q2_buffer.get_index()][24])
        self.Q1_buffer.init_hist()
        self.Q2_buffer.init_hist()
        self.update_sources()

    def update_sources(self):
        self.beam_source.data.update({
            "beam_values": self.beam_values_buffer.buffer,
            "Q1_buffer": self.Q1_buffer.buffer,
            "Q2_buffer": self.Q2_buffer.buffer,
        })
        self.Q_source.data.update({
            "Q1_hist": self.Q1_buffer.hist,
            "Q1_hist_bins": self.Q1_buffer.hist_bin_centers,
            "Q2_hist": self.Q2_buffer.hist,
            "Q2_hist_bins": self.Q2_buffer.hist_bin_centers,
        })

    def stream(self):
        if not hasattr(self, 'n'):
            self.n = 0
        self.n += 1
        self.step()
        self.beam_plot.title.text = f"N={self.n}"

    def initialize_widgets(self):
        pn.extension(design="material", sizing_mode="stretch_width")
        self.slider_m1 = pn.widgets.FloatSlider(
            value=self.prior_params.m1, start=0, end=2, step=0.05, name="M1"
        )
        self.slider_m1.param.watch(self.on_m1_change, 'value')

        self.slider_m2 = pn.widgets.FloatSlider(
            value=self.prior_params.m2, start=0, end=2, step=0.05, name="M2"
        )
        self.slider_m2.param.watch(self.on_m2_change, 'value')

        self.slider_m3 = pn.widgets.FloatSlider(
            value=self.prior_params.m3, start=0, end=2, step=0.05, name="M3"
        )
        self.slider_m3.param.watch(self.on_m3_change, 'value')

        self.slider_width = pn.widgets.FloatSlider(
            value=self.prior_params.width, start=0, end=1, step=0.01, name="Width"
        )
        self.slider_width.param.watch(self.on_width_change, 'value')

        self.callback = pn.state.add_periodic_callback(
            self.stream, self.callback_period, start=False
        )

        self.slider_speed = pn.widgets.FloatSlider(
            value=self.callback_period, start=1, end=500, step=10, name="Refresh Rate (ms)"
        )
        self.slider_speed.param.watch(self.on_speed_change, 'value')

        self.reset_button = pn.widgets.Button(name="Reset")
        self.reset_button.on_click(self.reset)

        self.pause_button = pn.widgets.Toggle(
            name="Start/Stop", value=False, button_type="primary"
        )
        self.pause_button.link(self.callback, bidirectional=True, value="running")

    def on_m1_change(self, event):
        self.prior_params.m1 = event.new

    def on_m2_change(self, event):
        self.prior_params.m2 = event.new

    def on_m3_change(self, event):
        self.prior_params.m3 = event.new

    def on_width_change(self, event):
        self.prior_params.width = event.new

    def on_speed_change(self, event):
        self.callback.period = event.new

    def reset(self, event):
        self.prior_params = self.reset_params()
        self.slider_m1.value = self.prior_params.m1
        self.slider_m2.value = self.prior_params.m2
        self.slider_m3.value = self.prior_params.m3
        self.slider_width.value = self.prior_params.width
        self.slider_speed.value = self.callback_period
        self.n = 0
        if not self.callback.running:
            self.callback.start()

    def setup_plots(self):
        self.beam_plot = plotting.figure(
            x_axis_label="Beam Element",
            y_axis_label="Value",
            title="N=0",
            y_range=[0, 1200],
            width=500,
            height=400,
        )
        self.beam_plot.multi_line(
            xs="beam_indices",
            ys="beam_values",
            line_color='blue',
            line_width=0.5,
            alpha=0.2,
            source=self.beam_source,
        )
        self.beam_plot.scatter(
            x="Q1_element", y="Q1_buffer", color='red', marker='x', source=self.beam_source
        )
        self.beam_plot.scatter(
            x="Q2_element", y="Q2_buffer", color='red', marker='x', source=self.beam_source
        )

        self.Q1_plot = plotting.figure(
            x_axis_label="Q1", title="Q1", width=300, height=400, x_range=[0, 1200]
        )
        self.Q1_plot.step(
            x="Q1_hist_bins",
            y="Q1_hist",
            line_color='blue',
            mode="center",
            line_width=2,
            source=self.Q_source,
        )

        self.Q2_plot = plotting.figure(
            x_axis_label="Q2", title="Q2", width=300, height=400, x_range=[0, 1200]
        )
        self.Q2_plot.step(
            x="Q2_hist_bins",
            y="Q2_hist",
            line_color='blue',
            mode="center",
            line_width=2,
            source=self.Q_source,
        )

    def setup_template(self):
        sliders = pn.Column(
            "### Prior Parameters",
            self.slider_m1,
            self.slider_m2,
            self.slider_m3,
            self.slider_width,
            pn.layout.Divider(),
            "### Playback Controls",
            self.slider_speed,
            pn.Row(self.reset_button, self.pause_button),
        )

        self.template = pn.template.MaterialTemplate(
            site="UM-Bridge App",
            title="MUQ Beam",
            header_background=PRIMARY_COLOR,
            sidebar=[sliders],
            main=[pn.Row(self.beam_plot, self.Q1_plot, self.Q2_plot)],
        )

    def serve(self):
        pn.serve(self.template)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Umbridge Panel App.')
    parser.add_argument('--url', type=str, default='http://localhost:4243',
                        help='The URL at which the model is running.')
    args = parser.parse_args()

    app = UmbridgePanelApp(url=args.url)
    app.serve()
