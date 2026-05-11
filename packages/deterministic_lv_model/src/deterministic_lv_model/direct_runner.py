"""Run the deterministic Lotka-Volterra model via Python."""

from mrp import Environment

from deterministic_lv_model import Deterministic_LV_Model

model_inputs = {
    "seed": 123,
    "max_time": 15,
    "a": 1.0,
    "b": 1.0,
    "x0": 1.0,
    "y0": 0.5,
    "obs_times": [1.1, 2.4, 3.9, 5.6, 7.5, 9.6, 11.9, 14.4],
    "obs_noise_mu": 0.0,
    "obs_noise_sigma": 0.5,
}

# This runs the model directly as a static method
results = Deterministic_LV_Model.simulate(model_inputs)

print("time | x observed | y observed")
print("-" * 45)
for t, observed_x, observed_y in zip(
    model_inputs["obs_times"],
    results["observed_x"],
    results["observed_y"],
):
    print(f"{t:>4.1f} | {observed_x:.6f} | {observed_y:.6f}")


# This runs the model via MRP,
# which generates files
env = Environment(
    {
        "input": model_inputs,
        "output": {"spec": "filesystem", "dir": "./output"},
    }
)
model = Deterministic_LV_Model(env=env)
model.run()
