import umbridge
import numpy as np
from pydantic import BaseModel

# Inspired by https://github.com/chi-feng/mcmc-demo

class DonutConfig(BaseModel):
    radius: float = 2.6
    sigma2: float = 0.033

class Donut(umbridge.Model):
    config: DonutConfig = DonutConfig()

    def __init__(self):
        super().__init__("posterior")

    def get_input_sizes(self, config):
        return [2]

    def get_output_sizes(self, config):
        return [1]

    def __call__(self, parameters, config):
        sigma2 = config.get('sigma2', self.config.sigma2)
        radius = config.get('radius', self.config.radius)                
        r = np.linalg.norm(parameters[0])
        return [[float(- (r - radius)**2 / sigma2) ]]

    def supports_evaluate(self):
        return True

    def gradient(self, out_wrt, in_wrt, parameters, sens, config):
        sigma2 = config.get('sigma2', self.config.sigma2)
        radius = config.get('radius', self.config.radius)        
        r = np.linalg.norm(parameters[0])
        if (r == 0):
            return [0,0]
        return [float(sens[0] * parameters[0][0] * (radius / r - 1) * 2 / sigma2),
                float(sens[0] * parameters[0][1] * (radius / r - 1) * 2 / sigma2)]

    def supports_gradient(self):
        return True

    def apply_jacobian(self, out_wrt, in_wrt, parameters, vec, config):
        sigma2 = config.get('sigma2', self.config.sigma2)
        radius = config.get('radius', self.config.radius)
        r = np.linalg.norm(parameters[0])
        if (r == 0):
            return [0]
        return [float(vec[0] * parameters[0][0] * (radius / r - 1) * 2 / sigma2
              + vec[1] * parameters[0][1] * (radius / r - 1) * 2 / sigma2)]

    def supports_apply_jacobian(self):
        return True

model = Donut()

umbridge.serve_models([model], 4243)