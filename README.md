# umbridge-workshop-2024
Repository for UM-Bridge 2024 workshop

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/krosenfeld-IDM/umbridge-workshop-2024)

## Usage

The Apps are in the `scripts/` directory:
1. `muq_beam/`: [cantilevered beam example](https://um-bridge-benchmarks.readthedocs.io/en/docs/forward-benchmarks/muq-beam-propagation.html
) (forward propagation)
2. `analytic_inverse/`: [analytic donut example](https://um-bridge-benchmarks.readthedocs.io/en/docs/inverse-benchmarks/analytic-donut.html) (sampling / inverse benchmark)
3. `deconvolution_1d/`: [1d deconvolution problem benchmark](https://um-bridge-benchmarks.readthedocs.io/en/docs/inverse-benchmarks/deconvolution-1d.html) (sampling)
4. `ew/`: England & Wales Measles model (application)

In each App directory you will want to:
1. Build and start the Docker container (see the local README.md for specific instruction), e.g.,:
```bash
docker build -t muq-beam .
docker run -it -p 4243:4243 muq-beam
```
2. Launch the app
```bash
python app.py
```


## Requirements
- Docker
- [uv](https://docs.astral.sh/uv/getting-started/)

## Setup with uv

0. Create and activate environment. 
```bash
uv venv
source .venv/bin/activate
```
1. Then install packages
```bash
uv pip install .
```