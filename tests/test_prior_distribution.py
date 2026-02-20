from math import exp

from numpy.random import SeedSequence
from scipy.stats import expon, lognorm, norm

from calibrationtools import IndependentPriors, SeedPrior, UniformPrior
from calibrationtools.prior_distribution import (
    ExponentialPrior,
    LogNormalPrior,
    NormalPrior,
)


def test_uniform_prior_sampling(
    seed_sequence: SeedSequence, seed_sequence_copy: SeedSequence
) -> None:
    prior = UniformPrior(param="x", min=0.0, max=10.0)
    samples = [s["x"] for s in list(prior.sample(5, seed_sequence))]

    # all in expected range and unique
    for sample in samples:
        assert 0.0 <= sample <= 10.0
    assert len(set(samples)) == len(samples)

    # reproducibility
    samples_copy = [s["x"] for s in list(prior.sample(5, seed_sequence_copy))]
    assert samples == samples_copy


def test_uniform_prior_probability() -> None:
    prior = UniformPrior(param="x", min=0.0, max=10.0)

    assert prior.probability_density({"x": 5.0}) == 1.0 / 10.0
    assert prior.probability_density({"x": -1.0}) == 0.0
    assert prior.probability_density({"x": 11.0}) == 0.0


def test_seed_prior_sampling(
    seed_sequence: SeedSequence, seed_sequence_copy: SeedSequence
) -> None:
    prior = SeedPrior(param="seed")
    samples = [s["seed"] for s in list(prior.sample(5, seed_sequence))]

    # all unique
    assert len(set(samples)) == len(samples)

    # reproducibility
    samples_copy = [
        s["seed"] for s in list(prior.sample(5, seed_sequence_copy))
    ]
    assert samples == samples_copy


def test_seed_prior_probability() -> None:
    prior = SeedPrior(param="seed")

    assert prior.probability_density({"seed": 123456}) == 1.0
    assert prior.probability_density({"not_seed": 123456}) == 0.0


def test_normal_prior_sampling(seed_sequence: SeedSequence) -> None:
    prior = NormalPrior(param="x", mean=0.0, std_dev=1.0)
    samples = [s["x"] for s in prior.sample(5, seed_sequence)]

    assert all(isinstance(v, float) for v in samples)


def test_normal_prior_probability() -> None:
    prior = NormalPrior(param="x", mean=0.0, std_dev=1.0)

    assert prior.probability_density({"x": 0.0}) == norm.pdf(
        0.0, loc=0.0, scale=1.0
    )
    assert prior.probability_density({"x": 2.0}) == norm.pdf(
        2.0, loc=0.0, scale=1.0
    )


def test_lognormal_prior_sampling(seed_sequence: SeedSequence) -> None:
    prior = LogNormalPrior(param="x", mean=0.0, std_dev=0.25)
    samples = [s["x"] for s in prior.sample(5, seed_sequence)]

    assert all(v > 0 for v in samples)


def test_lognormal_prior_probability() -> None:
    prior = LogNormalPrior(param="x", mean=0.0, std_dev=0.5)
    expected = lognorm.pdf(1.0, s=0.5, scale=exp(0.0))

    assert prior.probability_density({"x": 1.0}) == expected


def test_exponential_prior_sampling(seed_sequence: SeedSequence) -> None:
    prior = ExponentialPrior(param="x", rate=2.0)
    samples = [s["x"] for s in prior.sample(5, seed_sequence)]

    assert all(v >= 0 for v in samples)


def test_exponential_prior_probability() -> None:
    prior = ExponentialPrior(param="x", rate=2.0)

    assert prior.probability_density({"x": 0.5}) == expon.pdf(0.5, scale=0.5)


def test_independent_priors_sampling(
    seed_sequence: SeedSequence, seed_sequence_copy: SeedSequence
) -> None:
    prior1 = UniformPrior(param="x", min=0.0, max=10.0)
    prior2 = UniformPrior(param="y", min=20.0, max=100.0)
    indep_prior = IndependentPriors([prior1, prior2])

    samples = list(indep_prior.sample(5, seed_sequence))
    x_values = [s["x"] for s in samples]
    y_values = [s["y"] for s in samples]

    for x in x_values:
        assert 0.0 <= x <= 10.0
    for y in y_values:
        assert 20.0 <= y <= 100.0
    assert len(set(x_values)) == len(x_values)
    assert len(set(y_values)) == len(y_values)

    # reproducibility
    samples_copy = list(indep_prior.sample(5, seed_sequence_copy))
    x_values_copy = [s["x"] for s in samples_copy]
    y_values_copy = [s["y"] for s in samples_copy]
    assert x_values == x_values_copy
    assert y_values == y_values_copy


def test_independent_priors_probability() -> None:
    prior1 = UniformPrior(param="x", min=0.0, max=10.0)
    prior2 = UniformPrior(param="y", min=20.0, max=100.0)
    indep_prior = IndependentPriors([prior1, prior2])

    assert indep_prior.probability_density({"x": 5.0, "y": 50.0}) == (
        1.0 / 10.0
    ) * (1.0 / 80.0)
    assert indep_prior.probability_density({"x": -1.0, "y": 50.0}) == 0.0
    assert indep_prior.probability_density({"x": 5.0, "y": 10.0}) == 0.0
