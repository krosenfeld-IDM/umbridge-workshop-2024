""" server_funnel.py
https://github.com/UM-Bridge/benchmarks/blob/main/benchmarks/analytic-funnel/server.py
"""
import umbridge
import numpy as np
from pydantic import BaseModel

# "funnel" distribution from Neal, Radford M. 2003. “Slice Sampling.” Annals of Statistics 31 (3): 705–67
# Implementation inspired by https://github.com/chi-feng/mcmc-demo

class FunnelConfig(BaseModel):
    m0: float = 0
    s0: float = 3
    m1: float = 0

class Funnel(umbridge.Model):
    config: FunnelConfig = FunnelConfig()

    def __init__(self):
        super().__init__("posterior")

    def get_input_sizes(self, config):
        return [2]

    def get_output_sizes(self, config):
        return [1]

    def __call__(self, parameters, config):
        def f(x, m, s):
            return -0.5 * np.log(2 * np.pi) - np.log(s) - 0.5 * ((x-m)/s)**2
        m0 = config.get("m0", self.config.m0)
        s0 = config.get("s0", self.config.s0)
        m1 = config.get("m1", self.config.m1)
        s1 = np.exp(parameters[0][0] / 2)
        return [[ float(f(parameters[0][0], m0, s0) + f(parameters[0][1], m1, s1)) ]]

    def supports_evaluate(self):
        return True

    def gradient(self, out_wrt, in_wrt, parameters, sens, config):
        return [float(self.apply_jacobian(out_wrt, in_wrt, parameters, [sens[0], 0], config)[0]),
                float(self.apply_jacobian(out_wrt, in_wrt, parameters, [0, sens[0]], config)[0])]

    def supports_gradient(self):
        return True

    def apply_jacobian(self, out_wrt, in_wrt, parameters, vec, config):
        def f(x, m, s):
            return -0.5 * np.log(2 * np.pi) - np.log(s) - 0.5 * ((x-m)/s)**2
        def dfdx(x, m, s):
            return -(x-m) / (s**2)
        def dfds(x, m, s):
            return ((x-m)**2 - (s**2)) / (s**3)
        m0 = config.get("m0", self.config.m0)
        s0 = config.get("s0", self.config.s0)
        m1 = config.get("m1", self.config.m1)
        s1 = np.exp(parameters[0][0] / 2)

        return [float(vec[1] * dfdx(parameters[0][1], m1, s1)
              + vec[0] * (dfdx(parameters[0][0], m0, s0) + .5 * s1 * dfds(parameters[0][1], m1, s1)))]

    def supports_apply_jacobian(self):
        return True

model = Funnel()

umbridge.serve_models([model], 4243)