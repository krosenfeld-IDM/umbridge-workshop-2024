"""
https://www.pymc.io/projects/examples/en/stable/howto/sampling_callback.html
TODO: fix reset
"""
import traceback
import argparse
import pymc as pm
import panel as pn
from bokeh import models
from bokeh import plotting
import viz_umbridge as vu
from umbridge.pymc import UmbridgeOp


class PanelPymcApp(vu.UmbridgePanelApp):

    def __init__(self, url, model_name="posterior"):
        self.config = {'m0': 0, 's0': 3, 'm1': 0}
        self.op = UmbridgeOp(args.url, "posterior", config=self.config)
        self.input_dim = self.op.umbridge_model.get_input_sizes()[0]

        super().__init__(url, model_name)

        self.reset_params()
        self.initialize_buffers()
        self.initialize_plot_sources()
        self.initialize_widgets()
        self.setup_plots()
        self.setup_template()

    def reset_params(self):
        super().reset_params()
        self.tuned = False
        self.stepping = 0
        self.config['m0'] = 0
        self.config['s0'] = 3
        self.config['m1'] = 0

    def initialize_widgets(self):
        super().initialize_widgets()

        for key, value in self.config.items():
            slider = pn.widgets.FloatSlider(name=key, start=-10, end=10, value=value)
            # add attribute to self
            setattr(self, f'on_{key}_change', lambda event, key=key: self.config.update({key: event.new}))
            slider.param.watch(getattr(self, f'on_{key}_change'), 'value')
            self.sliders.append(slider)

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
        # don't step again until we're done
        if self.stepping:
            return
        
        with pm.Model() as model:
            self.stepping = 1 # set stepping flag

            try:
                # UM-Bridge models with a single 1D output implementing a PDF
                # may be used as a PyMC density that in turn may be sampled
                posterior = pm.DensityDist('posterior',logp=self.op,shape=self.input_dim)

                # map_estimate = pm.find_MAP()
                # print(f"MAP estimate of posterior is {map_estimate['posterior']}")
                trace = pm.sample(tune=10, draws=50, cores=1, return_inferencedata=False, step=pm.NUTS())

                # update the data buffers
                for points in trace.points():
                    for i, point in enumerate(points['posterior']):
                        self.data_buffers[f"var_{i}"].add(point)

                self.update_plot_sources()

            except Exception as e:
                traceback.print_exc()

            self.stepping = 0 # reset stepping flag

    def stream(self):
        super().stream()
        self.plots[0].title.text = f"N={self.n}"        


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Umbridge Panel App.')
    parser.add_argument('--url', type=str, default='http://localhost:4243',
                        help='The URL at which the model is running.')
    args = parser.parse_args()

    app = PanelPymcApp(url=args.url)

    app.serve()
