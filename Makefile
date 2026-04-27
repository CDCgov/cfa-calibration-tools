.DEFAULT_GOAL := help

UV ?= uv
PYTHON ?= python
PRE_COMMIT ?= pre-commit
DOCKER ?= docker

EXAMPLE_PACKAGE ?= example-model
DOCKER_IMAGE ?= cfa-calibration-tools-example-model-python:latest
DOCKERFILE ?= packages/example_model/Dockerfile

MRP_CONFIG ?= example_model.mrp.toml
MRP_DOCKER_CONFIG ?= example_model.mrp.docker.toml
MRP_CLOUD_CONFIG ?= example_model.mrp.cloud.toml

ARTIFACTS_DIR ?= ./artifacts
MAX_CONCURRENT_SIMULATIONS ?= 25

MRP_ARGS ?=
CALIBRATE_ARGS ?=
CLOUD_ARGS ?=
TEST_ARGS ?=
RUFF_ARGS ?=
TY_ARGS ?=

SESSION_SLUG ?=
IMAGE_TAG ?=

ifneq ($(strip $(SESSION_SLUG)),)
SESSION_FLAG := --session-slug $(SESSION_SLUG)
endif

ifneq ($(strip $(IMAGE_TAG)),)
IMAGE_TAG_FLAG := --image-tag $(IMAGE_TAG)
endif

.PHONY: help
help: ## Show available targets.
	@awk 'BEGIN {FS = ":.*##"; printf "Usage: make <target> [VAR=value]\n\nTargets:\n"} /^[a-zA-Z0-9_.-]+:.*##/ {printf "  %-24s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

.PHONY: sync
sync: ## Sync all workspace packages and extras.
	$(UV) sync --all-packages --all-extras

.PHONY: sync-cloud
sync-cloud: ## Sync workspace packages with CloudOps dependencies.
	$(UV) sync --all-packages --group cloudops

.PHONY: lock
lock: ## Update uv.lock.
	$(UV) lock

.PHONY: test
test: ## Run the full pytest suite. Pass TEST_ARGS='...' for filters.
	$(UV) run pytest $(TEST_ARGS)

.PHONY: test-core
test-core: ## Run tests for the calibrationtools package.
	$(UV) run pytest tests $(TEST_ARGS)

.PHONY: test-example
test-example: ## Run tests for the bundled example model package.
	$(UV) run pytest packages/example_model/tests $(TEST_ARGS)

.PHONY: lint
lint: ## Run ruff checks.
	$(UV) run ruff check --line-length 79 . $(RUFF_ARGS)

.PHONY: format
format: ## Format Python code with ruff.
	$(UV) run ruff format --line-length 79 . $(RUFF_ARGS)

.PHONY: format-check
format-check: ## Check Python formatting.
	$(UV) run ruff format --check --line-length 79 . $(RUFF_ARGS)

.PHONY: typecheck
typecheck: ## Run ty type checks.
	$(UV) run ty check --ignore=unresolved-import $(TY_ARGS)

.PHONY: precommit
precommit: ## Run all configured pre-commit hooks.
	$(PRE_COMMIT) run --all-files

.PHONY: check
check: lint format-check typecheck test ## Run lint, format check, type check, and tests.

.PHONY: docker-build
docker-build: ## Build the example model Docker image.
	$(DOCKER) build -t $(DOCKER_IMAGE) -f $(DOCKERFILE) .

.PHONY: mrp
mrp: ## Run the default inline MRP config. Pass MRP_ARGS='--input seed=42'.
	$(UV) run --package $(EXAMPLE_PACKAGE) mrp run $(MRP_CONFIG) $(MRP_ARGS)

.PHONY: mrp-docker
mrp-docker: docker-build ## Build the image and run the Docker-backed MRP config.
	$(UV) run --package $(EXAMPLE_PACKAGE) mrp run $(MRP_DOCKER_CONFIG) $(MRP_ARGS)

.PHONY: mrp-cloud
mrp-cloud: ## Run the cloud executor MRP config locally. Usually use calibrate-cloud instead.
	$(UV) run --group cloudops --package $(EXAMPLE_PACKAGE) mrp run $(MRP_CLOUD_CONFIG) $(MRP_ARGS)

.PHONY: calibrate
calibrate: ## Run the local in-process example calibration.
	$(UV) run $(PYTHON) -m example_model.calibrate $(CALIBRATE_ARGS)

.PHONY: calibrate-docker
calibrate-docker: docker-build ## Run example calibration through the Docker-backed MRP config.
	$(UV) run $(PYTHON) -m example_model.calibrate --docker $(CALIBRATE_ARGS)

.PHONY: calibrate-cloud
calibrate-cloud: ## Run cloud-backed example calibration. Pass CLOUD_ARGS='...'.
	$(UV) run --group cloudops $(PYTHON) -m example_model.calibrate --cloud $(CLOUD_ARGS)

.PHONY: calibrate-cloud-auto
calibrate-cloud-auto: ## Run cloud calibration with auto-size and progress output.
	$(UV) run --group cloudops $(PYTHON) -m example_model.calibrate --cloud --auto-size --print-task-progress --print-task-durations --artifacts-dir $(ARTIFACTS_DIR) $(CLOUD_ARGS)

.PHONY: benchmark
benchmark: ## Compare serial and parallel example calibration execution.
	$(UV) run $(PYTHON) -m example_model.benchmark

.PHONY: cloud-list
cloud-list: ## List cloud resources. Optional: SESSION_SLUG=... IMAGE_TAG=...
	$(UV) run --group cloudops $(PYTHON) -m example_model.cloud_cleanup --config $(MRP_CLOUD_CONFIG) --list $(SESSION_FLAG) $(IMAGE_TAG_FLAG)

.PHONY: cloud-cleanup-plan
cloud-cleanup-plan: ## Show the cleanup plan for SESSION_SLUG=... without deleting.
	@test -n "$(SESSION_SLUG)" || (echo "SESSION_SLUG is required"; exit 1)
	$(UV) run --group cloudops $(PYTHON) -m example_model.cloud_cleanup --config $(MRP_CLOUD_CONFIG) $(SESSION_FLAG) $(IMAGE_TAG_FLAG)

.PHONY: cloud-cleanup-delete
cloud-cleanup-delete: ## Delete cloud resources for SESSION_SLUG=... Optional: IMAGE_TAG=...
	@test -n "$(SESSION_SLUG)" || (echo "SESSION_SLUG is required"; exit 1)
	$(UV) run --group cloudops $(PYTHON) -m example_model.cloud_cleanup --config $(MRP_CLOUD_CONFIG) $(SESSION_FLAG) $(IMAGE_TAG_FLAG) --yes
