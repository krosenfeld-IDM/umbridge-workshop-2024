import os
import argparse
import numpy as np
import pandas as pd
import panel as pn
from bokeh import models
from bokeh import plotting
import viz_umbridge as vu
import umbridge
from bokeh.palettes import Reds256

from laser_model.england_wales.scenario import get_scenario


class EWApp(vu.UmbridgePanelApp):
    def __init__(self, url, model_name="forward"):
        super().__init__(url, 'England&Wales Measles', model_name)

        self.config = {}
        self.n_nodes: int = None
        self.reset_params()

        self.umbridge_model =  umbridge.HTTPModel(url, "forward")

        self.initialize_plot_sources()
        self.initialize_wave_buffer()
        self.initialize_widgets()
        self.setup_plots()
        self.setup_template()

    def reset(self, event):
        super().reset(event)
        self.umbridge_model([[]], config={'reset': True})

    def reset_params(self):
        super().reset_params()
        self.wave_radius = 30
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
            "cases": np.zeros(len(scenario))
        })
        self.plot_wave_source = models.ColumnDataSource({
            "x": [],
            "y": []
        })
        self.n_nodes = len(scenario)

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

        wave_plot = plotting.figure(
            x_axis_label="Distance",
            y_axis_label="Phase",
            title="Traveling Wave",
            width=500,
            height=500,
            y_range=(-np.pi, np.pi)
        )
        wave_plot.scatter(x="x", y="y", source=self.plot_wave_source, alpha=0.5)
        self.plots += [wave_plot]

    def step(self):
        # increment the step counter
        self.n += 1

        # steps the UMBridge model        
        res = self.umbridge_model([[]])

        # update the plot sources
        self.plot_node_source.data.update({'prevalence': res[0]})
        self.plot_node_source.data.update({'cases': res[1]})
        self.wave_buffer.add(res[1])

        if self.n % 10 == 0:
            self.calculate_wave('London', self.wave_radius)

    def stream(self):
        super().stream()
        self.plots[0].title.text = f"N={self.tick}"

    def initialize_wave_buffer(self, buffer_size:int = 26*4):
        self.wave_buffer = vu.FixedSizeObjectBuffer(buffer_size, placeholder=self.n_nodes*[0])


    def calculate_wave(self, ref, radius):
        data = pd.DataFrame(self.plot_node_source.data)
        data.reset_index(inplace=True)
        cases = np.array(self.wave_buffer.buffer)
        ref_cwt, _ = vu.measles.calc_Ws(cases[:, data[data.name == ref].index].flatten())
        distances = self.calculate_distances(ref)
        ind = np.nonzero(distances < radius)[0]
        x = np.zeros(len(ind))
        y = np.zeros(len(ind))
        for cnt, i in enumerate(ind):
            row = data.loc[i]
            distance = distances.iloc[i]
            if row.name == ref:
                continue
            cwt, frequencies = vu.measles.calc_Ws(cases[:, i].flatten())
            diff = np.conjugate(ref_cwt)*cwt
            ind = np.where(np.logical_and(frequencies < 1/(1.5 * 26), frequencies > 1 / (3 * 26)))            
            diff = diff[ind[0], :]
            x[cnt] = distance
            y[cnt] = np.angle(np.mean(diff))
        self.plot_wave_source.data.update({'x': x, 'y': y})

    @pn.cache
    def calculate_distances(self, ref: str = 'London'):
        data = pd.DataFrame(self.plot_node_source.data)
        place = data.loc[ref]
        deg2km = 111.32
        distances = np.sqrt((place.x - data.x)**2 + (place.y - data.y)**2) * deg2km
        return distances

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
    # for i in range(20):
    #     app.step()

    # app.calculate_wave('London', 30)

    app.serve()

    print("pause")