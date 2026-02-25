# ABC-SMC interface
## Algorithm for ABC-SMC
1. Sample $n$ particles from the joint prior distribution $\pi(\theta)$ and store as current population set $\mathbb{A}$
2. Initialize an empty proposed particle population $\mathbb{B}$
3. For each generation $g$ specified in the tolerance error array $\vec\epsilon$
    a. While $\mathbb{B}$ has fewer than $n$ particles:
        i. Sample a particle $j$ from $\mathbb{A}$
        ii. Perturb selected particle to make $\hat\theta_j$
        iii. If $\pi(\hat\theta_j) > 0$, continue, otherwise go to 3.a.i
        iv. Run model with particle $\hat\theta_j$
        v. Collect outputs and calculate distance $d_j$
        vi. If $d_j<\epsilon_g$,
            1. Calculate weight $w_j$ based on $\mathbb{A}$ and $\pi(\theta)$
            2. Add $\hat\theta_j$ with weight $w_j$ to population $\mathbb{B}$
        vii. Go to 3.a
    b. Archive population $\mathbb{A}$
    c. Normalize weights of population $\mathbb{B}$ and adapt perturbation variance
    d. Set $\mathbb{A}$ to $\mathbb{B}$ and initialize new population $\mathbb{B}$

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
