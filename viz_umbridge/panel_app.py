import panel as pn

PRIMARY_COLOR = "#780078"  # UM-Bridge purple
SECONDARY_COLOR = "#F5A91E"  # UM-Bridge yellow

class UmbridgePanelApp:
    def __init__(self, url, model_name="posterior"):
        self.url = url
        self.model_name = model_name
        self.title = "Umbridge App"
        self.callback_period = None
        self.n = None
        
        self.plots = []
        self.sliders = []

    def reset_params(self):
        self.callback_period = 50
        self.n = 0

    def initialize_buffers(self):
        pass

    def initialize_plot_sources(self):
        pass

    def initialize_widgets(self):
        pn.extension(design="material", sizing_mode="stretch_width")

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

    def on_speed_change(self, event):
        self.callback.period = event.new

    def update_plot_sources(self):
        pass

    def step(self):
        pass

    def stream(self):
        self.n += 1
        self.step()    

    def setup_template(self, sliders: list =None):
        sliders = (
            ["### Prior Parameters ###"] 
            + self.sliders 
            + [
                pn.layout.Divider(),
                "### Playback Controls",
                self.slider_speed,
                pn.Row(self.reset_button, self.pause_button),
            ]
        )
        sliders = pn.Column(*sliders)

        self.template = pn.template.MaterialTemplate(
            site="Umbridge Panel App",
            title=self.title,
            header_background=PRIMARY_COLOR,
            sidebar=[sliders],
            main=[pn.Row(*self.plots)],
        )


    def reset(self, event):
        self.params = self.reset_params()
        self.n = 0
        if not self.callback.running:
            self.callback.start()

    def serve(self):
        pn.serve(self.template)        