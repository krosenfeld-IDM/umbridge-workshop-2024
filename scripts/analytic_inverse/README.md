# README

## Usage
Build specific server (e.g., `server_funnel.py`) we pass in the model name (e.g., `funnel`):
```bash
docker build --build-arg MODEL=funnel -t analytic-funnel .
docker run -it -p 4243:4243 analytic-funnel
```

## More info
Fast/analytic inverse problems w/ gradient:
- https://um-bridge-benchmarks.readthedocs.io/en/docs/inverse-benchmarks/analytic-donut.html
```bash
docker run -it -p 4243:4243 linusseelinger/benchmark-analytic-donut
```
- https://um-bridge-benchmarks.readthedocs.io/en/docs/inverse-benchmarks/analytic-funnel.html
```bash
docker run -it -p 4243:4243 linusseelinger/benchmark-analytic-funnel
```

w/o gradient:
- https://um-bridge-benchmarks.readthedocs.io/en/docs/inverse-benchmarks/analytic-banana.html

To start:
```bash
docker run -it -p 4243:4243 linusseelinger/benchmark-analytic-banana
```