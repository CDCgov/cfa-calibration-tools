## Potential structures to keep from abc-smc
- Particles  this structure can be the same for now (particles to params for model output), but this will eventually need to work with MRP
- Generation  reframe as a population of particles
- Particle updater  operates at population level
- Variance adapter  operates at population level
- Prior distribution  element of experiment controller
- Perturbation kernels  element of experiment controller in concert with variance adapter
- Spawn RNG  we do want a method for handling RNGs, this is one option


## Orchestrator script design
```python run_calibration.py
#| evaluate: false
def particles_to_params():

def outputs_to_distance():

## This will be substituted by MRP
def model_runner():


sampler = ABCSampler(
    generation_particle_count = 100,
    tolerance_values = [10, 5, 1],
    priors = params_priors,
    perturbations,
    particles_to_params,
    outputs_to_distance,
    target_data = data_df,
    model_runner = model_runner,
    seed = 12354)

sampler.run()
posterior_particles = sampler.get_posterior_particles()
```

## Algorithm design
```python abc_sampler.py
#| evaluate: false
class ABCSampler:
    ''' Combines functionality from abc_smc.ParticleUpdater and abc_smc.Experiment'''
    def __init__(
            generation_particle_count, # Number of particles to accept for each generation
            tolerance_values, # Tolerance threshold of acceptance for distacne in each step, length is the number of steps in the SMC algorithm
            priors, # Dictionary contianing distribution information
            perturbations, # Dictionary controlling methods (independent, variance controlling, etc) and parameter kernels
            particles_to_params, # Function to turn particles into parameter sets for the runner
            outputs_to_distance, # Fucntion to turn model outputs into distances given target data
            target_data, # Observed data to be used in calibration
            model_runner, # Protocol to turn parameter sets into model outputs
            seed # Seed for overall calibration runner
    ):
        ## Validation and initialization here

    def sample_particles_from_priors(self, n=None) -> ParticlePopulation:
        if not n:
            n = self.generation_particle_count
        population = ParticlePopulation()
        for i in range(n)
            particle = sample_from_priors()
            population.add(particle)
    return population

    def run():
        previous_population = self.sample_particles_from_priors()

        for generation in range(len(self.tolerance_values)):
            current_population = ParticlePopulation()

            while current_population.size < self.generation_particle_count:
                particle = previous_population.sample_particle()
                perturbed_particle = self.perturb_particle(particle, previous_population)

                self.particles_to_params(perturbed_particle)
                self.model_runner.run()
                err = self.outputs_to_distance()

                if err < self.tolerance_values[generation]:
                    perturbed_particle.weight = self.calculate_weight(perturbed_particle)
                    current_population.add(perturbed_particle)

            self.previous_population_archive[generation] = self.previous_population
            previous_population = current_population.normalize_weights()

        self.posterior_population = current_population
```
