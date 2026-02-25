# ABC-SMC interface
## Algorithm for ABC-SMC
1. Sample $n$ particles from the joint prior distribution $\pi(\theta)$ and store as current population set $\mathbb{A}_0$
2. Initialize an empty proposed particle population $\mathbb{B}_0$
3. For each generation $g$ specified in the tolerance error array $\vec\epsilon$
    1. While $\mathbb{B}_g$ has fewer than $n$ particles:
        1. Sample a particle $j$ from $\mathbb{A}_g$
        2. Perturb selected particle to make $\hat\theta_j$
        3. If $\pi(\hat\theta_j) > 0$, continue, otherwise go to 3.i.a
        4. Run model with particle $\hat\theta_j$
        5. Collect outputs and calculate distance $d_j$
        6. If $d_j<\epsilon_g$,
            1. Calculate weight $w_j$ based on $\mathbb{A}_g$ and $\pi(\theta)$
            2. Add $\hat\theta_j$ with weight $w_j$ to population $\mathbb{B}_g$
        7. Go to 3.i
    2. Handle population changes
        1. Archive population $\mathbb{A}_g$
        2. Normalize weights of population $\mathbb{B}_g$ and adapt perturbation variance
        3. Set $\mathbb{A}_{g+1}$ equal to the normalized proposed population

## Orchestrator script example
```python run_calibration.py
#| evaluate: false

# Create the prior distribution list
P = IndependentPriors(
    [
        UniformPrior("param1", 0, 1),
        NormalPrior("param2", 0, 1),
        LogNormalPrior("param3", 0, 1),
        ExponentialPrior("param4", 1),
        SeedPrior("seed"),
    ]
)

# Make list of independent kernels for the parameter perturbations
K = IndependentKernels(
    [
        MultivariateNormalKernel(
            [p.params[0] for p in P.priors if not isinstance(p, SeedPrior)],
        ),
        SeedKernel("seed"),
    ]
)

# Set the variance adapter for altering perturbation kernel steps sizes across SMC generations
V = AdaptMultivariateNormalVariance()

# Import or define the model runner
class SomeModelRunner:
    # The model runner must contain a `simulate` method
    def simulate(self, params):

# Function to convert a particle into parameter set for the model runner
def particles_to_params(particle):
    return particle

# Function to convert outputs from the model runner to a distance measure for use in algorithm step 3.a.v
def outputs_to_distance(model_output, target_data):
    return abs(model_output - target_data)

# Define the smapler using assembled components
sampler = ABCSampler(
    generation_particle_count=500,
    tolerance_values=[5.0, 0.5, 0.1],
    priors=P,
    perturbation_kernel=K,
    variance_adapter=V,
    particles_to_params=particles_to_params,
    outputs_to_distance=outputs_to_distance,
    target_data=0.5,
    model_runner=SomeModelRunner(),
    seed=123,
)

# run the calibration routine
sampler.run()
```
