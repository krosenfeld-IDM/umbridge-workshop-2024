"""
https://um-bridge-benchmarks.readthedocs.io/en/docs/inverse-benchmarks/deconvolution-1d.html
"""
import traceback
import argparse
import umbridge
import pymc as pm
import numpy as np
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
            pass
            # raise pm.exceptions.SamplingError("Sampling stopped by user.")
        
def reset_config():
    return {'delta': 0.01}
        
class PanelPymcApp(vu.UmbridgePanelApp):

    def __init__(self, url, model_name="posterior", reset_config=None):

        super().__init__(url, '1D Deconvolution', model_name)
        
        self.config = {}
        self.reset_params()
        self.input_dim = len(self.get_solution())
        self.sampler_callback = StopSamplingCallback(self)

        self.initialize_buffers()
        self.initialize_plot_sources()
        self.initialize_widgets()
        self.setup_plots()
        self.setup_template()

    @pn.cache
    def get_solution(self):
        return umbridge.HTTPModel(self.url, "Deconvolution1D_ExactSolution")([[]])[0]

    def set_op(self):
        # Set up an pytensor op connecting to UM-Bridge model
        self.op = UmbridgeOp(self.url, self.select.value)        

    def reset_params(self):
        super().reset_params()
        self.stepping = False
        self.start = None
        for k,v in reset_config().items():
            self.config[k] = v
        if hasattr(self, 'select'):
            self.op = UmbridgeOp(self.url, self.select.value, config=self.config)
        else:
            self.op = UmbridgeOp(self.url, 'Deconvolution1D_Gaussian', config=self.config)

    def reset(self, event):
        super().reset(event)
        self.stepping = False
        self.start = None
        self.initialize_buffers()
        for k, v in self.sliders.items():
            if k in self.config:
                v.value = self.config[k]      

    def setup_plots(self):
        sample_plot = plotting.figure(title="Samples", width=400, height=400)
        sample_plot.line(x=np.arange(self.input_dim), y=self.get_solution(), color='red')
        sample_plot.line(x="x", y="mean", source=self.plot_source, color='black')
        sample_plot.line(x="x", y="lower", source=self.plot_source, color='blue')
        sample_plot.line(x="x", y="upper", source=self.plot_source, color='blue')

        self.plots += [sample_plot]
        
    def initialize_widgets(self):
        super().initialize_widgets()

        menu_items = umbridge.supported_models(args.url)
        self.select = pn.widgets.Select(name='Select', options=menu_items)
        self.sliders['prior'] = self.select

        for key, value in self.config.items():
            slider = pn.widgets.FloatSlider(name=key, start=-10, end=10, value=value)
            setattr(self, f'on_{key}_change', lambda event, key=key: self.config.update({key: event.new}))
            slider.param.watch(getattr(self, f'on_{key}_change'), 'value')
            self.sliders[f'{key}'] = slider

    def initialize_buffers(self, buffer_size=1000):
        self.buffer = vu.FixedSizeObjectBuffer(buffer_size, placeholder=np.full(self.input_dim, np.nan))

    def initialize_plot_sources(self):
        self.plot_source = models.ColumnDataSource({"x": np.arange(self.input_dim), "mean": np.zeros(self.input_dim), 
                                                    "lower": np.zeros(self.input_dim), "upper": np.zeros(self.input_dim)})

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
                    trace = pm.sample(tune=0, draws=5, initvals={'posterior': np.array(self.get_solution())}, **kwargs)
                else:
                    trace = pm.sample(tune=0, draws=5, start=self.start, **kwargs)

                self.start = trace.point(-1)

                for points in trace.points():
                    self.buffer.add(points['posterior'])

                traces = np.array(self.buffer.buffer)
                self.plot_source.data.update({'mean': np.nanmean(traces, axis=0), 
                                              'lower': np.nanpercentile(traces, 2.5, axis=0),
                                              'upper': np.nanpercentile(traces, 97.5, axis=0)})

            except pm.exceptions.SamplingError:
                traceback.print_exc()
                print("Sampling was stopped by the user.")

            except Exception as e:
                traceback.print_exc()

            finally:
                self.stepping = False

        return True        


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Umbridge Panel App.')
    parser.add_argument('--url', type=str, default='http://localhost:4245',
                        help='The URL at which the model is running.')
    parser.add_argument('--model', type=str, default='donut',
                        help='The name of the model to be used.')
    args = parser.parse_args()
        
    app = PanelPymcApp(url=args.url)

    app.serve()            