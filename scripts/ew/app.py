import os
import argparse
import numpy as np
import pandas as pd
import panel as pn
from collections import deque
from bokeh import models
from bokeh import plotting
import viz_umbridge as vu
import umbridge
from bokeh.palettes import Reds256

from laser_model.england_wales.scenario import get_scenario
from laser_model.england_wales.params import get_parameters

class EWApp(vu.UmbridgePanelApp):
    def __init__(self, url, model_name="forward"):
        super().__init__(url, 'England&Wales Measles', model_name)

        self.config = {}
        self.n_nodes: int = None
        self.param_dict = {'beta':{'start':0, 'end':50, 'step':1}, 'seasonality':{'start':0, 'end':0.3, 'step':0.02}, 
                           'demog_scale':{'start':0.1, 'end':1.5, 'step':0.05}, 
                           'mixing_scale':{'start':-4, 'end':-2, 'step':0.5}, 
                           'distance_exponent':{'start':1.0, 'end':2.0, 'step':0.1}}
        self.reset_params()

        self.umbridge_model =  umbridge.HTTPModel(url, "forward")

        self.initialize_plot_sources()
        self.initialize_buffers()
        self.initialize_widgets()
        self.setup_plots()
        self.setup_template()

    def reset(self, event):
        super().reset(event)
        self.umbridge_model([[]], config={'reset': True})
        for k, v in self.sliders.items():
            if k in self.config:
                v.value = self.config[k]        
        self.time_ts_list.clear()
        self.prev_ts_list.clear()        

    def reset_params(self):
        super().reset_params()
        self.wave_radius = 30
        self.callback_period = 50
        for p in self.param_dict.keys():
            self.config[p] = get_parameters({})[p]
            if p == 'mixing_scale':
                self.config[p] = np.log10(self.config[p])

    def initialize_widgets(self):

        super().initialize_widgets()

        self.wave_button = pn.widgets.Toggle(
            name="start/stop wave", value=False, button_type="default"
        )     

        for key, value in self.config.items():
            slider = pn.widgets.FloatSlider(name=key, value=value, **self.param_dict[key])
            setattr(self, f'on_{key}_change', lambda event, key=key: self.config.update({key: event.new}))
            slider.param.watch(getattr(self, f'on_{key}_change'), 'value')
            self.sliders[f'{key}'] = slider


    def initialize_plot_sources(self):
        scenario = get_scenario()
        self.plot_node_source = models.ColumnDataSource({
            "name": scenario.index,
            "x": scenario.Long,
            "y": scenario.Lat,
            "size": 0.03 * np.sqrt(scenario.population),
            "population": scenario.population,
            "births": scenario.births,
            "prevalence": np.zeros(len(scenario))
                            })
        self.plot_wave_source = models.ColumnDataSource({
            "x": [],
            "y": []
        })
        self.ts_source = models.ColumnDataSource({"time": np.arange(0, 10 * 26), "prevalence": np.zeros(10 * 26)})
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

        prev_ts = plotting.figure(x_axis_label="Time (years)", y_axis_label="Prevalence (%)", width=500, height=200)
        prev_ts.line(x="time", y="prevalence", source=self.ts_source, color="red") 
        self.plots += [prev_ts]     

    def step(self):
        # increment the step counter
        self.n += 1

        # steps the UMBridge model        
        res = self.umbridge_model([[]], config=self.config)

        # update the plot sources
        self.plot_node_source.data.update({'prevalence': res[0]})
        self.plot_node_source.data.update({'cases': res[1]})
        self.wave_buffer.add(res[1])

        self.time_ts_list.append(self.n / 26.0)
        self.prev_ts_list.append(100 * res[2][0])  # (prev in %)        
        self.ts_source.data = {"time": list(self.time_ts_list), "prevalence": list(self.prev_ts_list)}

        if self.wave_button.value & (self.n % 26 == 0):
            self.calculate_wave('London', self.wave_radius)

    def stream(self):
        super().stream()
        self.plots[0].title.text = f"N={self.tick}"

    def initialize_buffers(self, buffer_size:int = 26*4):
        self.time_ts_list = deque()
        self.prev_ts_list = deque()
        self.wave_buffer = vu.FixedSizeObjectBuffer(buffer_size, placeholder=self.n_nodes*[0])


    def calculate_wave(self, ref, radius):
        data = pd.DataFrame(self.plot_node_source.data)
        data.reset_index(inplace=True)
        cases = np.array(self.wave_buffer.buffer)
        ref_cwt, _ = vu.measles.calc_Ws(cases[:, data[data.name == ref].index].flatten())
        # ref_cwt = np.conjugate(ref_cwt)
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
            diff = ref_cwt*np.conj(cwt)
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
    
    def setup_template(self, sliders: list =None):
        sliders = (
            ["### Parameters ###"] 
            + [s for s in self.sliders.values()]
            + [
                pn.layout.Divider(),
                "### Playback Controls",
                self.slider_speed,
                pn.Row(self.reset_button, self.pause_button),
                self.wave_button, 
            ]
        )
        sliders = pn.Column(*sliders)

        self.template = pn.template.MaterialTemplate(
            site="UM-Bridge App",
            title=self.title,
            header_background=vu.PRIMARY_COLOR,
            sidebar=[sliders],
            main=[pn.Row(*self.plots[:2]), self.plots[2]],
        )    

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