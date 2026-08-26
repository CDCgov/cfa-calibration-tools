# Design Reference

---

## Table of Contents

1. [Context Orchestrator](01-context-orchestrator.md)
2. [Calibrator Construction](02-calibrator-construction.md)
3. [Simulation System and DAG Stages](03-simulation-system-and-dag-stages.md)
4. [Particle Type and Overlay Semantics](04-particle-type-and-overlay-semantics.md)
5. [ParticlePopulation, Weights, and Product](05-particlepopulation-weights-and-product.md)
6. [PerturbationKernel and Density Convention](06-perturbationkernel-and-density-convention.md)
   - [PerturbationKernel Trait](06-perturbationkernel-and-density-convention.md#perturbationkernel-trait)
   - [Perturbation Policy](06-perturbationkernel-and-density-convention.md#perturbation-policy-on-calibrationbuilder)
   - [Between-Stage Adaptation Lifecycle](06-perturbationkernel-and-density-convention.md#between-stage-adaptation-lifecycle)
7. [NestedSuffixParser and ParticleError](07-nestedsuffixparser-and-particleerror.md)
8. [ScoreCalculator and Target Builder](08-scorecalculator-and-target-builder.md)
9. [ScoreAcceptanceCriterion](09-scoreacceptancecriterion.md)
10. [Counterfactual Model Construction](10-counterfactual-model-construction.md)
11. [Fingerprinting and Caching Strategy](11-fingerprinting-and-caching-strategy.md)
12. [ModelRunner and Python Interop](12-modelrunner-and-python-interop.md)
13. [Seeds and RNG Discipline](13-seeds-and-rng-discipline.md)
14. [Error Propagation and Stage Resumability](14-error-propagation-and-stage-resumability.md)
15. [ABC Rejection Sampling Execution](15-abc-rejection-sampling-execution.md)
    - [15.1 Execution Modes](15-abc-rejection-sampling-execution.md#151-execution-modes)
16. [Runtime Execution](16-runtime-execution.md)
    - [16.1 RunBuilder](16-runtime-execution.md#161-runbuilder)
    - [16.2 SimulationResult](16-runtime-execution.md#162-simulationresult)
    - [16.3 Failed Particle Rerun](16-runtime-execution.md#163-failed-particle-rerun)
    - [16.4 Calibration Comparison](16-runtime-execution.md#164-calibration-comparison)
17. [End-to-end example](17-end-to-end-example.md)
