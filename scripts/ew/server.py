import umbridge
import numpy as np
from laser_model.england_wales.model import EnglandWalesModel
from laser_model.england_wales.params import get_parameters
from laser_model.england_wales.scenario import get_scenario
from laser_model.mixing import init_gravity_diffusion

# from . import analyze as ana

class ForwardModel(umbridge.Model):
    def __init__(self, name: str ='forward', config: dict = None):
        super().__init__(name)
        self.reset()
        self.config = config if config is not None else {}
        self.model = EnglandWalesModel(parameters=self.params, scenario=get_scenario())
        self.model.metrics = []
        self.reset()

    def reset(self):
        self.tick = 0
        self.params = get_parameters({})

    def reset_state(self):
        self.reset()
        self.model.init_state(get_scenario(), self.params)

    def get_input_sizes(self, config):
        return [0]

    def get_output_sizes(self, config):
        return 2*[len(self.model.nodes)] + [1]
    
    def step(self):
        self.model.step(self.tick)
        self.tick += 1
    
    def __call__(self, parameters:list=None, config:dict=None):
        if config is not None:
            mix_flag = False
            if config.get('reset', False):
                self.reset_state()
            for p,v in config.items():
                if p == 'mixing_scale':
                    v = np.power(10, v)
                if (p in self.model.params) and (self.model.params[p]) != v:
                    self.model.params[p] = v
                    if p in ['distance_exponent', 'mixing_scale']:
                        mix_flag = True
            if mix_flag:
                self.model.params['mixing'] = init_gravity_diffusion(get_scenario(), self.model.params.mixing_scale, self.model.params.distance_exponent)
        self.step()

        # package results
        prevalence = self.model.nodes.states[1] / self.model.nodes.states.sum(axis=0)
        cases = self.model.nodes.states[1] # number of cases is just infected because of 2 week time step
        total_prevalence = self.model.nodes.states[1].sum() / self.model.nodes.states.sum()
        return [prevalence.tolist(), cases.tolist(), [total_prevalence]]

    def supports_evaluate(self):
        return True

umbridge.serve_models(
    [ForwardModel()], 4243
)