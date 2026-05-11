"""Plot deterministic Lotka-Volterra trajectories."""

import matplotlib.pyplot as plt
import numpy as np

from deterministic_lv_model import Deterministic_LV_Model

max_time = 15.0
dt = 0.01
obs_times = np.arange(0.0, max_time, dt).tolist()

model_inputs = {
    "seed": 123,
    "max_time": max_time,
    "a": 1,
    "b": 1,
    "x0": 1.0,
    "y0": 0.5,
    "obs_times": obs_times,
    "obs_noise_mu": 0.0,
    "obs_noise_sigma": 0.0,
    "stop_x_threshold": 1e6,
}

results = Deterministic_LV_Model.simulate(model_inputs)

plt.figure(figsize=(5, 5))
plt.plot(
    results["times"], results["observed_x"], linestyle="-", label="x (prey)"
)
plt.plot(
    results["times"],
    results["observed_y"],
    linestyle="--",
    label="y (predator)",
)
plt.xlabel("Time")
plt.ylabel("Population")
plt.xlim(0, 15)
plt.xticks([0, 5, 10, 15])
plt.ylim(0, 4)
plt.yticks([0, 1, 2, 3, 4])
plt.title(
    "Deterministic Lotka-Volterra model \n compare to Fig. 1(a) in Toni et al. 2009"
)
plt.legend()
plt.tight_layout()
plt.show()
