"""
"""
import os
import arviz as az
import argparse
import umbridge
import numpy as np
import matplotlib.pyplot as plt

import pymc as pm
from pytensor import tensor as pt
from pytensor.gradient import verify_grad # noqa: F401
from umbridge.pymc import UmbridgeOp

# Change to directory of this script
os.chdir(os.path.dirname(__file__))

# Read URL from command line argument
parser = argparse.ArgumentParser(description='Minimal HTTP model demo.')
parser.add_argument('--url', metavar='url', type=str, default='http://localhost:4243',
                    help='the URL at which the model is running, for example http://localhost:4243 (default: http://localhost:4243)')
args = parser.parse_args()
print(f"Connecting to host URL {args.url}")

# Print modelssupported by server
print(umbridge.supported_models(args.url))

# Set up an pytensor op connecting to UM-Bridge model
config = {'m0': 0, 's0': 3, 'm1': 0}
op = UmbridgeOp(args.url, "posterior", config=config)

print(op.umbridge_model.get_output_sizes())
print(op.umbridge_model.get_input_sizes())

# # Define input parameter
input_dim = op.umbridge_model.get_input_sizes()[0]
input_val = np.random.rand(input_dim)

# Evaluate model with input parameter
op_application = op(pt.as_tensor_variable(input_val))
print(f"Model output: {op_application.eval()}")

# # Verify gradient
# print("Check model's gradient against numerical gradient. This requires an UM-Bridge model with gradient support.")
# verify_grad(op, [input_val], rng = np.random.default_rng())

with pm.Model() as model:
    # UM-Bridge models with a single 1D output implementing a PDF
    # may be used as a PyMC density that in turn may be sampled
    posterior = pm.DensityDist('posterior',logp=op,shape=input_dim)

    map_estimate = pm.find_MAP()
    print(f"MAP estimate of posterior is {map_estimate['posterior']}")

    inferencedata = pm.sample(tune=100,draws=400,cores=1)
    az.plot_pair(inferencedata)
    plt.savefig("pymc_example.png")


    