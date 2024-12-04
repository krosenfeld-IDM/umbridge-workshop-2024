"""
https://www.pymc.io/projects/examples/en/stable/howto/sampling_callback.html
"""

import traceback
import argparse
import pymc as pm
import panel as pn
from pytensor import tensor as pt
from bokeh import models
from bokeh import plotting
import viz_umbridge as vu
from umbridge.pymc import UmbridgeOp

class StopSamplingCallback(pm.callbacks.Callback):
    def __init__(self, app):
        self.app = app

    def __call__(self, trace, draw, **kwargs):
        if not self.app.callback.running:
            raise pm.exceptions.SamplingError("Sampling stopped by user.")

class PanelPymcApp(vu.UmbridgePanelApp):

    def __init__(self, url, model_name="posterior", reset_config=None):

        super().__init__(url, model_name)

        setattr(self, 'reset_config', reset_config)

        self.config = {}
        self.reset_params()

        self.op = UmbridgeOp(args.url, "posterior", config=self.config)
        self.input_dim = self.op.umbridge_model.get_input_sizes()[0]
        self.sampler_callback = StopSamplingCallback(self)

        self.initialize_buffers()
        self.initialize_plot_sources()
        self.initialize_widgets()
        self.setup_plots()
        self.setup_template()

    def reset_params(self):
        super().reset_params()
        self.stepping = False
        self.start = None
        for k,v in self.reset_config().items():
            self.config[k] = v

    def reset(self, event):
        super().reset(event)
        self.stepping = False
        self.start = None
        for k, v in self.sliders.items():
            v.value = self.config[k]

    def initialize_widgets(self):
        super().initialize_widgets()

        for key, value in self.config.items():
            slider = pn.widgets.FloatSlider(name=key, start=-10, end=10, value=value)
            setattr(self, f'on_{key}_change', lambda event, key=key: self.config.update({key: event.new}))
            slider.param.watch(getattr(self, f'on_{key}_change'), 'value')
            self.sliders[f'{key}'] = slider

    def initialize_buffers(self, buffer_size=500):
        self.data_buffers = {f"var_{i}": vu.FixedSizeFloatBuffer(buffer_size, placeholder=i) for i in range(self.input_dim)}

    def initialize_plot_sources(self):
        self.plot_source = models.ColumnDataSource({f"var_{i}":[] for i in range(self.input_dim)})

    def update_plot_sources(self):
        self.plot_source.data.update({
            f"var_{i}": self.data_buffers[f"var_{i}"].buffer
            for i in range(self.input_dim)
        })

    def setup_plots(self):
        sample_plot = plotting.figure(title="Samples", width=400, height=400)
        sample_plot.scatter(x="var_0", y="var_1", source=self.plot_source, 
                            color='blue', marker='o', alpha=0.5)
        self.plots += [sample_plot]

    def step(self):
        if self.stepping:
            return False

        self.stepping = True

        with pm.Model() as model:
            try:
                posterior = pm.DensityDist('posterior', logp=self.op, shape=self.input_dim)
                kwargs = {
                    'step': pm.Metropolis(),
                    'return_inferencedata': False,
                    'cores': 1,
                    'callback': self.sampler_callback
                }
                if self.start is None:
                    trace = pm.sample(tune=0, draws=50, **kwargs)
                else:
                    trace = pm.sample(tune=0, draws=50, start=self.start, **kwargs)

                self.start = trace.point(-1)

                for points in trace.points():
                    for i, point in enumerate(points['posterior']):
                        self.data_buffers[f"var_{i}"].add(point)

                self.update_plot_sources()

            except pm.exceptions.SamplingError:
                traceback.print_exc()
                print("Sampling was stopped by the user.")

            except Exception as e:
                traceback.print_exc()

            finally:
                self.stepping = False

        return True

    def stream(self):
        if self.stepping:
            return
        status = super().stream()
        if status:
            self.plots[0].title.text = f"N={self.n}"        


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Umbridge Panel App.')
    parser.add_argument('--url', type=str, default='http://localhost:4243',
                        help='The URL at which the model is running.')
    parser.add_argument('--model', type=str, default='donut',
                        help='The name of the model to be used.')
    args = parser.parse_args()

    if args.model == 'donut':
        def default_config():
            return {'radius': 2.6, 'sigma2': 0.033}
    else:
        def default_config():
            return {'m0': 0, 's0': 3, 'm1': 0}
        
    app = PanelPymcApp(url=args.url, reset_config=default_config)

    app.serve()