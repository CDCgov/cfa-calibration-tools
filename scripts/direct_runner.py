"""Run the example branching process via Python."""

from mrp import Environment

from example_model import Binom_BP_Model

# This runs the model directly as a static method
results = Binom_BP_Model.simulate(
    {
        "seed": 123,
        "max_gen": 15,
        "n": 3,
        "p": 0.5,
        "max_infect": 500,
    },
)

print("Generation | Population")
print("-" * 25)
for gen, pop in enumerate(results):
    print(f"{gen:>10} | {pop}")


# This runs the model via MRP,
# which generates files
env = Environment(
    {
        "input": {
            "seed": 123,
            "max_gen": 15,
            "n": 3,
            "p": 0.5,
            "max_infect": 500,
        },
        "output": {"spec": "filesystem", "dir": "./output"},
    }
)
model = Binom_BP_Model(env=env)
model.run()
