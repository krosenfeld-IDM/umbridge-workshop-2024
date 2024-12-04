import pymc as pm

class Callback:
    def __init__(self, every=10):
        self.traces = {}
        self.every = every
        self.multitrace = None

    def __call__(self, trace, draw):
        # if draw.tuning:
        #     return
        if len(trace) % self.every == 0:
            self.traces[draw.chain] = trace
            self.multitrace = pm.backends.base.MultiTrace(list(self.traces.values()))

