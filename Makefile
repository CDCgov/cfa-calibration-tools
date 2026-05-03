.DEFAULT_GOAL := help

UV ?= uv
PYTHON ?= python
PRE_COMMIT ?= pre-commit
DOCKER ?= docker

EXAMPLE_PACKAGE ?= example-model
DOCKER_IMAGE ?= cfa-calibration-tools-example-model-python:latest
DOCKERFILE ?= packages/example_model/Dockerfile

MRP_CONFIG ?= packages/example_model/src/example_model/example_model.mrp.toml
MRP_DOCKER_CONFIG ?= packages/example_model/src/example_model/example_model.mrp.docker.toml
CLOUD_CONFIG ?= packages/example_model/src/example_model/example_model.cloud_config.toml

# Passthrough variables append raw CLI arguments after target-owned defaults.
# Example: make calibrate-cloud CALIBRATE_ARGS='--artifacts-dir /tmp/run'
MRP_ARGS ?=
CALIBRATE_ARGS ?=
TEST_ARGS ?=
RUFF_ARGS ?=
TY_ARGS ?=

# CLOUD_USER intentionally defaults to the current shell user. The cleanup
# targets therefore operate on the caller's own cloud sessions unless a
# different CLOUD_USER is provided explicitly.
SESSION_ID ?=
CLOUD_USER ?= $(shell id -un 2>/dev/null || $(PYTHON) -c 'import getpass; print(getpass.getuser())')
DRY_RUN ?=
IMAGE_TAG ?=

ifneq ($(strip $(SESSION_ID)),)
SESSION_ID_FLAG := --session-id $(SESSION_ID)
endif

ifneq ($(strip $(CLOUD_USER)),)
USER_FLAG := --user $(CLOUD_USER)
endif

ifneq ($(strip $(DRY_RUN)),)
DRY_RUN_FLAG := --dry-run
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
check: precommit test ## Run precommit, and tests.

.PHONY: docker-build
docker-build: ## Build the example model Docker image.
	$(DOCKER) build -t $(DOCKER_IMAGE) -f $(DOCKERFILE) .

.PHONY: mrp
mrp: ## Run the default inline MRP config. Pass MRP_ARGS='--input seed=42'.
	$(UV) run --package $(EXAMPLE_PACKAGE) mrp run $(MRP_CONFIG) $(MRP_ARGS)

.PHONY: mrp-docker
mrp-docker: docker-build ## Build the image and run the Docker-backed MRP config.
	$(UV) run --package $(EXAMPLE_PACKAGE) mrp run $(MRP_DOCKER_CONFIG) $(MRP_ARGS)

.PHONY: calibrate
calibrate: ## Run local calibration. Pass CALIBRATE_ARGS='...'.
	$(UV) run $(PYTHON) -m example_model.calibrate $(CALIBRATE_ARGS)

.PHONY: calibrate-docker
calibrate-docker: docker-build ## Run Docker-backed calibration. Pass CALIBRATE_ARGS='...'.
	$(UV) run $(PYTHON) -m example_model.calibrate --docker $(CALIBRATE_ARGS)

.PHONY: calibrate-cloud
calibrate-cloud: ## Run cloud-backed calibration. Pass CALIBRATE_ARGS='...'.
	$(UV) run --group cloudops $(PYTHON) -m example_model.calibrate --cloud --cloud-config $(CLOUD_CONFIG) $(CALIBRATE_ARGS)

.PHONY: calibrate-cloud-auto
calibrate-cloud-auto: ## Run cloud calibration with auto-size/progress. Pass CALIBRATE_ARGS='...'.
	$(UV) run --group cloudops $(PYTHON) -m example_model.calibrate --cloud --cloud-config $(CLOUD_CONFIG) --auto-size --print-task-progress $(CALIBRATE_ARGS)

.PHONY: benchmark
benchmark: ## Compare serial and parallel example calibration execution.
	$(UV) run $(PYTHON) -m example_model.benchmark

.PHONY: cloud-list
cloud-list: ## List cloud resources. Optional: SESSION_ID=... CLOUD_USER=... IMAGE_TAG=...
	$(UV) run --group cloudops $(PYTHON) -m calibrationtools.cloud.cleanup --cloud-config $(CLOUD_CONFIG) --list $(SESSION_ID_FLAG) $(USER_FLAG) $(IMAGE_TAG_FLAG)

.PHONY: cloud-cleanup-plan
cloud-cleanup-plan: ## Compatibility alias: preview cleanup for SESSION_ID=...
	$(MAKE) cloud-cleanup SESSION_ID="$(SESSION_ID)" IMAGE_TAG="$(IMAGE_TAG)" DRY_RUN=1

.PHONY: cloud-cleanup-delete
cloud-cleanup-delete: ## Compatibility alias: delete cleanup for SESSION_ID=...
	$(MAKE) cloud-cleanup SESSION_ID="$(SESSION_ID)" IMAGE_TAG="$(IMAGE_TAG)"

.PHONY: cloud-cleanup
cloud-cleanup: ## Delete cloud resources for SESSION_ID=... Optional: DRY_RUN=1 IMAGE_TAG=...
	@test -n "$(SESSION_ID)" || (echo "SESSION_ID is required"; exit 1)
	$(UV) run --group cloudops $(PYTHON) -m calibrationtools.cloud.cleanup --cloud-config $(CLOUD_CONFIG) $(SESSION_ID_FLAG) $(IMAGE_TAG_FLAG) $(DRY_RUN_FLAG)

.PHONY: cloud-cleanup-user
cloud-cleanup-user: ## Delete all cloud sessions for CLOUD_USER; defaults to current user. Optional: DRY_RUN=1 IMAGE_TAG=...
	@test -n "$(CLOUD_USER)" || (echo "CLOUD_USER is required"; exit 1)
	$(UV) run --group cloudops $(PYTHON) -m calibrationtools.cloud.cleanup --cloud-config $(CLOUD_CONFIG) --all-sessions-for-user $(USER_FLAG) $(IMAGE_TAG_FLAG) $(DRY_RUN_FLAG)
