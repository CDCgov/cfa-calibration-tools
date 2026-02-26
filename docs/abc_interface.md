# ABC-SMC interface
## Algorithm for ABC-SMC
1. Specify the joint prior distribution $\pi(\theta)$
2. Initialize an empty proposed particle population $\mathbb{B}_0$
3. For each generation $g$ specified in the tolerance error array $\vec\epsilon$
    1. Initialize an empty population $\mathbb{B}_g$
    2. While $\mathbb{B}_g$ has fewer than $n$ particles:
        1. Propose a particle.
            1. If $g=0$, sample a parameter set from $\pi(\theta)$ and store as particle $\hat\theta_j$,
            2. Otherwise, sample a particle $j$ from $\mathbb{A}_{g-1}$ and perturb the selected particle to make $\hat\theta_j$
        2. If $\pi(\hat\theta_j) > 0$, continue, otherwise go to 3.i.a
        3. Run model with particle $\hat\theta_j$
        4. Collect outputs and calculate distance $d_j$
        5. If $d_j<\epsilon_g$,
            1. If $g=0$, set weight $w_j=1.0$. Otherwise, calculate weight $w_j$ based on $\mathbb{A}_{g-1}$ and $\pi(\theta)$
            2. Add $\hat\theta_j$ with weight $w_j$ to population $\mathbb{B}_g$
        7. Go to 3.i
    3. Handle population changes
        1. Normalize weights of proposed population $\mathbb{B}_g$ and adapt perturbation variance
        2. Set current population $\mathbb{A}_{g}$ equal to the normalized proposed population

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
