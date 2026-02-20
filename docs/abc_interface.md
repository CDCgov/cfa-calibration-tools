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
            priors, # Dictionary containing distribution information
            perturbations, # Dictionary controlling methods (variance adapter and kernels) and parameter kernels
            variance_adapter, # Object specifeid for controlling the change in variance across SMC steps
            particles_to_params, # Function to turn particles into parameter sets for the runner
            outputs_to_distance, # Fucntion to turn model outputs into distances given target data
            target_data, # Observed data to be used in calibration
            model_runner, # Protocol to turn parameter sets into model outputs
            seed # Seed for overall calibration runner
    ):
        ## Validation and initialization here

        ## Init updater
        self.updater = _ParticleUpdater(perturbations, variance_adapter)

    def run(self):
        previous_population = self.sample_particles_from_priors()

        for generation in range(len(self.tolerance_values)):
            print(
                f"Running generation {generation + 1} with tolerance {self.tolerance_values[generation]}... previous population size is {previous_population.size}"
            )
            current_population = ParticlePopulation()  # Inits a new population
            self.set_population(
                previous_population
            )  # sets `all_particles` to the previous population

            # Rejection sampling algorithm
            attempts = 0
            while current_population.size < self.generation_particle_count:
                if attempts % 100 == 0:
                    print(
                        f"Attempt {attempts}... current population size is {current_population.size}. Acceptance rate is {current_population.size / attempts if attempts > 0 else 0:.4f}",
                        end="\r",
                    )
                attempts += 1
                # Create the parameter inputs for the runner by sampling perturbed value from previous population
                particle = self.sample_particle()
                perturbed_particle = self.perturb_particle(particle)
                params = self.particles_to_params(perturbed_particle)

                # Generate the distance metric from model run
                outputs = self.model_runner.simulate(params)
                err = self.outputs_to_distance(outputs, self.target_data)

                # Add the particle to the population if accepted
                if err < self.tolerance_values[generation]:
                    perturbed_particle.weight = self.calculate_weight(
                        perturbed_particle
                    )
                    current_population.add(perturbed_particle)

            # Archive the previous generation population and make new population for next step
            self.previous_population_archive.update(
                {generation: previous_population}
            )
            current_population.normalize_weights()
            previous_population = current_population

        # Store posterior particle population
        self.posterior_population = current_population

    def set_population(self, population: ParticlePopulation):
        self._updater.set_particle_population(population)

    def sample_particles_from_priors(self, n=None) -> ParticlePopulation:
        '''Return a particle from the prior distribution'''
        if not n:
            n = self.generation_particle_count
        population = ParticlePopulation()
        for _ in range(n):
            particle_state = self.priors.sample_state()
            population.add(Particle(state=particle_state))
        return population

    def perturb_particle(self, particle: Particle) -> Particle:
        return self._updater.perturb_particle(particle)

    def sample_particle(self) -> Particle:
        return self._updater.sample_particle()

    def calculate_weight(self, particle) -> float:
        return self._updater.calculate_weight(particle)

    def get_posterior_particles(self) -> ParticlePopulation:
        return self.posterior_population

```
