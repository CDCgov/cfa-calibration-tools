# Deterministic Lotka-Volterra model

This python package implements the deterministic Lotka-Volterra model from Toni et al. 2009 and is intended to be used to recreate their application of ABC-SMC.

Source: Toni T, Welch D, Strelkowa N, Ipsen A, Stumpf MP. Approximate Bayesian computation scheme for parameter inference and model selection in dynamical systems. J R Soc Interface. 2009 Feb 6;6(31):187-202. doi: 10.1098/rsif.2008.0172. PMID: 19205079; PMCID: PMC2658655.

This README will describe multiple ways to run the model.

## Running the model
First open the Python interactive shell within the `uv` environment:

```bash
uv sync --all-packages
uv run python
```
The following two lines of code are sufficient to run the deterministic Lotka-Volterra model using the static method (i.e., calling `Deterministic_LV_Model.simulate()`).
```python
from deterministic_lv_model import Deterministic_LV_Model
Deterministic_LV_Model.simulate({"seed": 123, "max_time": 15, "a": 1.0, "b": 1.0, "x0": 1.0, "y0": 0.5, "obs_times": [1.1, 2.4, 3.9, 5.6, 7.5, 9.6, 11.9, 14.4], "obs_noise_mu": 0.0, "obs_noise_sigma": 0.5})
```
This should yield the following output:
```python
{'times': [1.1, 2.4, 3.9, 5.6, 7.5, 9.6, 11.9, 14.4], 'observed_x': [1.1246771238482753, 1.166751046707449, 1.1817395123483583, 0.7563737716738126, 2.0353883089084412, 1.0939384282697222, 0.2948374699905357, 2.008327213030393], 'observed_y': [0.5436012995980883, 1.4731030828379437, 1.3372496451894869, -0.17250775456526868, 1.2624073640868958, 1.3691942199186347, 1.1322475380937604, 0.9458138115125306]}
```
An equivalent approach is to read the model input from `defaults.json`. (The code here assumes working from the root of the repo.)

```python
import json
from deterministic_lv_model import Deterministic_LV_Model
model_inputs = json.load(open("./packages/deterministic_lv_model/defaults.json"))
Deterministic_LV_Model.simulate(model_inputs)
```
To use the MRP functionality, create an environment that specifies the inputs and use the `.run()` method:
```python
from mrp import Environment
from deterministic_lv_model import Deterministic_LV_Model
model_inputs = {"seed": 123, "max_time": 15, "a": 1.0, "b": 1.0, "x0": 1.0, "y0": 0.5, "obs_times": [1.1, 2.4, 3.9, 5.6, 7.5, 9.6, 11.9, 14.4], "obs_noise_mu": 0.0, "obs_noise_sigma": 0.5}
env = Environment({"input": model_inputs})
Deterministic_LV_Model(env).run()
```
The above examples are very similar to those included in `deterministic_lv_model/direct_runner.py`, which can be run (from the root of the repo) with the following:
```bash
uv sync --all-packages
uv run python -m deterministic_lv_model.direct_runner
```
Additionally, as described in the repo-level README, the model can be run as specified in the `deterministic_lv_model.mrp.toml`, which can be run as follows:
```bash
uv sync --all-packages
uv run mrp run deterministic_lv_model.mrp.toml
```
To generate a plot of the model output that can be compared to Figure 1(a) in Toni et al. 2009, run the `plot_runner.py` script:
```bash
uv sync --all-packages
uv run python -m deterministic_lv_model.plot_runner
```

## Running the calibration
To run the calibration example for this model, run

```bash
uv sync --all-packages
uv run python -m deterministic_lv_model.calibrate
```
