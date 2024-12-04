import os
import matplotlib.pyplot as plt
from laser_core.laserframe import LaserFrame
from matplotlib.figure import Figure

from laser_model.base import BaseComponent
from laser_model.england_wales.model import EnglandWalesModel
from laser_model.england_wales.params import get_parameters
from laser_model.england_wales.scenario import get_scenario

os.chdir(os.path.dirname(__file__))

class TotalInfectiousReporter(BaseComponent):
    def __init__(self, model, verbose: bool = False) -> None:
        super().__init__(model, verbose)
        print("Initializing the infectious reporter")

        # add the infectious counter
        if not hasattr(model, "reports"):
            model.reports = LaserFrame(1)
        model.reports.add_vector_property("infectious", length=model.params.nticks)

    def __call__(self, model, tick):
        # count the infecteds
        model.reports.infectious[tick] = model.nodes.states[1].sum()

        return

    def plot(self, fig: Figure = None):
        _fig = plt.figure(dpi=128) if fig is None else fig
        plt.plot(self.model.reports.infectious)
        plt.title("Infectious")
        plt.xlabel("Time (Bi-weekly)")
        yield
        return


# setup the model
scenario = get_scenario()
parameters = get_parameters({})
model = EnglandWalesModel(scenario, parameters)

# add the reporter
model.components += [TotalInfectiousReporter]

# and run
model.run()

# visualize the results
model.visualize(pdf=False)

print("done")