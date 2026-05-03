# CFA-CALIBRATION-TOOLS

This project is a calibration runner framework that can be installed and integrated with a given model repository. The framework will be comprised of a library of calibration algorithms and diagnostics, a model runner protocols for interfacing with inputs and outputs, and settings for execution via cloud infrastructure.

## Project Admin

CDC Center for Forecasting and Outbreak Analytics.

## Getting Started

This project uses `uv` for python venv management. Be sure to have `uv` [installed on your machine](https://docs.astral.sh/uv/getting-started/installation/). Start with the Makefile entrypoints:

```bash
make setup
make test
make example
make help
```

The equivalent underlying setup command is:

```{bash}
uv sync --all-packages --all-extras
```

## Running the example model

The default MRP config uses the inline runtime, so `mrp` calls the example
model in the current Python process.

```bash
make example-mrp
make example-mrp MRP_ARGS='--input seed=42 --input max_gen=10'
```

The equivalent raw commands are:

```bash
# Default parameters
uv run mrp run packages/example_model/src/example_model/example_model.mrp.toml

## Override parameters
uv run --package example-model mrp run packages/example_model/src/example_model/example_model.mrp.toml --input seed=42 --input max_gen=10
```

You can run `uv tool install cfa-mrp` to omit the `uv run`.

To run the Docker-backed config through the Makefile:

```bash
make mrp-docker
```

The equivalent raw commands are:

```bash
docker build -t cfa-calibration-tools-example-model-python:latest -f packages/example_model/Dockerfile .
uv run --package example-model mrp run packages/example_model/src/example_model/example_model.mrp.docker.toml
```

The Docker-backed config runs the container as your current host UID/GID so the bind-mounted `./output` directory stays writable. If you have a stale `output/` from an older run with different ownership, remove it before retrying.

## Running a calibration

The repository includes a complete calibration example for the bundled example model:

```bash
make example
make example CALIBRATE_ARGS='--artifacts-dir /tmp/run'
```

The equivalent raw command is:

```bash
uv run python -m example_model.calibrate
```

This runs the ABC-SMC calibration workflow defined in `packages/example_model/src/example_model/calibrate.py` and prints the posterior summary and diagnostics.
By default, calibration stages per-simulation inputs and outputs under
`./artifacts`, including paths like
`artifacts/input/generation-0/gen_0_particle_0_attempt_0.json` and
`artifacts/output/generation-0/gen_0_particle_0_attempt_0/output.csv`.
Use `--artifacts-dir path/to/artifacts` to choose another location, or
`--no-artifacts` to disable artifact staging for non-cloud local runs.

To run calibration through the Docker-backed MRP config:

```bash
make example-docker
```

The equivalent raw commands are:

```bash
docker build -t cfa-calibration-tools-example-model-python:latest -f packages/example_model/Dockerfile .
uv run python -m example_model.calibrate --docker
```

You can also route calibration through a specific MRP config with `uv run python -m example_model.calibrate --mrp-config path/to/config.toml`.

To compare serial and parallel execution for the same example, run:

```bash
make example-benchmark
```

The equivalent raw command is:

```bash
uv run python -m example_model.benchmark
```

## General Disclaimer

This repository was created for use by CDC programs to collaborate on public health related projects in support of the [CDC mission](https://www.cdc.gov/about/organization/mission.htm). GitHub is not hosted by the CDC, but is a third party website used by CDC and its partners to share information and collaborate on software. CDC use of GitHub does not imply an endorsement of any one particular service, product, or enterprise.

## Public Domain Standard Notice

This repository constitutes a work of the United States Government and is not
subject to domestic copyright protection under 17 USC § 105. This repository is in
the public domain within the United States, and copyright and related rights in
the work worldwide are waived through the [CC0 1.0 Universal public domain dedication](https://creativecommons.org/publicdomain/zero/1.0/).
All contributions to this repository will be released under the CC0 dedication. By
submitting a pull request you are agreeing to comply with this waiver of
copyright interest.

## License Standard Notice

This repository is licensed under ASL v2 or later.

This source code in this repository is free: you can redistribute it and/or modify it under
the terms of the Apache Software License version 2, or (at your option) any
later version.

This source code in this repository is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the Apache Software License for more details.

You should have received a copy of the Apache Software License along with this
program. If not, see <http://www.apache.org/licenses/LICENSE-2.0.html>

The source code forked from other open source projects will inherit its license.

## Privacy Standard Notice

This repository contains only non-sensitive, publicly available data and
information. All material and community participation is covered by the
[Disclaimer](https://github.com/CDCgov/template/blob/master/DISCLAIMER.md)
and [Code of Conduct](https://github.com/CDCgov/template/blob/master/code-of-conduct.md).
For more information about CDC's privacy policy, please visit [http://www.cdc.gov/other/privacy.html](https://www.cdc.gov/other/privacy.html).

## Contributing Standard Notice

Anyone is encouraged to contribute to the repository by [forking](https://help.github.com/articles/fork-a-repo)
and submitting a pull request. (If you are new to GitHub, you might start with a
[basic tutorial](https://help.github.com/articles/set-up-git).) By contributing
to this project, you grant a world-wide, royalty-free, perpetual, irrevocable,
non-exclusive, transferable license to all users under the terms of the
[Apache Software License v2](http://www.apache.org/licenses/LICENSE-2.0.html) or
later.

All comments, messages, pull requests, and other submissions received through
CDC including this GitHub page may be subject to applicable federal law, including but not limited to the Federal Records Act, and may be archived. Learn more at [http://www.cdc.gov/other/privacy.html](http://www.cdc.gov/other/privacy.html).

## Records Management Standard Notice

This repository is not a source of government records but is a copy to increase
collaboration and collaborative potential. All government records will be
published through the [CDC web site](http://www.cdc.gov).
