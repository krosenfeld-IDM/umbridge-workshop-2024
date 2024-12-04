import os
import argparse
import numpy as np
from bokeh import models
from bokeh import plotting
import viz_umbridge as vu
import umbridge
from bokeh.palettes import Reds256

from laser_model.england_wales.scenario import get_scenario


class EWApp(vu.UmbridgePanelApp):
    def __init__(self, url, model_name="forward"):
        super().__init__(url, model_name)

        self.config = {}

        self.reset_params()

        self.umbridge_model =  umbridge.HTTPModel(url, "forward")

        self.initialize_plot_sources()
        self.initialize_widgets()
        self.setup_plots()
        self.setup_template()

    def reset(self, event):
        super().reset(event)
        self.umbridge_model([[]], config={'reset': True})

    def reset_params(self):
        super().reset_params()
        self.callback_period = 50

    def initialize_plot_sources(self):
        scenario = get_scenario()
        self.plot_node_source = models.ColumnDataSource({
            "name": scenario.index,
            "x": scenario.Long,
            "y": scenario.Lat,
            "size": 0.03 * np.sqrt(scenario.population),
            "population": scenario.population,
            "births": scenario.births,
            "prevalence": np.zeros(len(scenario)),
        })

    def setup_plots(self):
        prev_plot = plotting.figure(
            x_axis_label="Longitude",
            y_axis_label="Latitude",
            title="Prevalence",
            width=500,
            height=500,
        )        
        prev_cmap = models.LogColorMapper(palette=Reds256[::-1], low=1e-4, high=0.01)
        prev_plot.scatter(x="x", y="y", size="size", color={"field": "prevalence", "transform": prev_cmap}, 
                             source=self.plot_node_source, alpha=0.5)

        self.plots += [prev_plot]

    def step(self):
        # increment the step counter
        self.n += 1

        # steps the UMBridge model        
        res = self.umbridge_model([[]])

        # update the plot sources
        self.plot_node_source.data.update({'prevalence': res[0]})

    def stream(self):
        super().stream()
        self.plots[0].title.text = f"N={self.tick}"     

if __name__ == "__main__":
    # Change to directory of this script
    os.chdir(os.path.dirname(__file__))

    # Read URL from command line argument
    parser = argparse.ArgumentParser(description='Minimal HTTP model demo.')
    parser.add_argument('--url', metavar='url', type=str, default='http://localhost:4243',
                        help='the URL at which the model is running, for example http://localhost:4243 (default: http://localhost:4243)')
    args = parser.parse_args()

    print(f"Connecting to host URL {args.url}")
    print(umbridge.supported_models(args.url))

    app = EWApp(args.url)

    app.serve()

    print("pause")