"""
docker run -it -p 4243:4243 linusseelinger/benchmark-deconvolution-1d:latest

References
- https://um-bridge-benchmarks.readthedocs.io/en/docs/inverse-benchmarks/deconvolution-1d.html

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
parser.add_argument('--sample', metavar='sample', type=bool, default=True)
args = parser.parse_args()
print(f"Connecting to host URL {args.url}")

# Print models supported by server
print(umbridge.supported_models(args.url))

# Get the exact solution
sol = umbridge.HTTPModel(args.url, "Deconvolution1D_ExactSolution")([[]])

# Set up an pytensor op connecting to UM-Bridge model
op = UmbridgeOp(args.url, "Deconvolution1D_Gaussian")

# Define input parameter
input_dim = len(sol[0])
init_vals = {'posterior': np.array(sol[0])} 

# Evaluate model with input parameter
op_application = op(pt.as_tensor_variable(init_vals['posterior']))
print(f"Model output: {op_application.eval()}")

# # # Verify gradient
# # print("Check model's gradient against numerical gradient. This requires an UM-Bridge model with gradient support.")
# # verify_grad(op, [input_val], rng = np.random.default_rng())

num_tune = 100
if args.sample:
    with pm.Model() as model:
        # UM-Bridge models with a single 1D output implementing a PDF
        # may be used as a PyMC density that in turn may be sampled
        posterior = pm.DensityDist('posterior',logp=op,shape=input_dim)

        # map_estimate = pm.find_MAP()
        # print(f"MAP estimate of posterior is {map_estimate['posterior']}")

        inferencedata = pm.sample(tune=num_tune,draws=800,cores=1, step=pm.NUTS(), initvals=init_vals, return_inferencedata=False)
        samples = inferencedata.get_values('posterior')
        np.save('pymc.npy', samples)

else:
    samples = np.load('pymc.npy')
    print('pause')

plt.figure()
plt.plot(sol[0], label="Exact Solution", color='k')
plt.plot(samples.mean(axis=0), label="Posterior Mean", color='C0')
plt.fill_between(np.arange(len(sol[0])), samples.mean(axis=0) - 2.96*samples.std(axis=0), samples.mean(axis=0) + 2.96*samples.std(axis=0), alpha=0.3, label="Posterior Std", color='C0')
plt.tight_layout()
plt.savefig("figure.png")

    