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
CLOUD_USER ?= $(or $(notdir $(lastword $(subst \, ,$(USER)))),$(firstword $(shell id -un 2>/dev/null || whoami 2>/dev/null || printf unknown)))
DRY_RUN ?=
IMAGE_TAG ?=
SKIP_ACR ?=

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

ifneq ($(strip $(SKIP_ACR)),)
SKIP_ACR_FLAG := --skip-acr
endif

.PHONY: help
help:
	@awk 'BEGIN {FS = ":.*## "; printf "Usage:\n  make <target> [VAR=value]\n"} /^##@ / {printf "\n%s:\n", substr($$0, 5); next} /^[a-zA-Z0-9_.-]+:.*## / {printf "  %-28s %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@printf '\nMore:\n  make help-vars\n  make help-cloud\n'

##@ Start here

.PHONY: setup
setup: sync ## Install/sync local development dependencies.

.PHONY: test
test: ## Run the full pytest suite. Pass TEST_ARGS='...' for filters.
	$(UV) run pytest $(TEST_ARGS)

.PHONY: example
example: calibrate ## Run the bundled calibration example locally.

.PHONY: cloud-run-auto
cloud-run-auto: calibrate-cloud-auto ## Run cloud calibration with auto-size/progress.

##@ Setup

.PHONY: sync
sync: ## Sync all workspace packages and extras.
	$(UV) sync --all-packages --all-extras

.PHONY: setup-cloud
setup-cloud: sync-cloud ## Install/sync local dependencies plus CloudOps.

.PHONY: sync-cloud
sync-cloud: ## Sync workspace packages with CloudOps dependencies.
	$(UV) sync --all-packages --group cloudops

.PHONY: lock
lock: ## Update uv.lock.
	$(UV) lock

##@ Quality

.PHONY: ci
ci: check ## Run the local CI-like gate.

.PHONY: fix
fix: ## Format Python code and then run ruff checks.
	$(MAKE) format
	$(MAKE) lint

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

##@ Example model

.PHONY: example-mrp
example-mrp: mrp ## Run the bundled MRP example.

.PHONY: example-docker
example-docker: calibrate-docker ## Build the image and run Docker-backed calibration.

.PHONY: example-benchmark
example-benchmark: benchmark ## Compare serial and parallel example calibration.

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

.PHONY: benchmark
benchmark: ## Compare serial and parallel example calibration execution.
	$(UV) run $(PYTHON) -m example_model.benchmark

##@ Cloud

.PHONY: cloud-run
cloud-run: calibrate-cloud ## Run cloud-backed calibration.

.PHONY: calibrate-cloud
calibrate-cloud: ## Run cloud-backed calibration. Pass CALIBRATE_ARGS='...'.
	$(UV) run --group cloudops $(PYTHON) -m example_model.calibrate --cloud --cloud-config $(CLOUD_CONFIG) $(CALIBRATE_ARGS)

.PHONY: calibrate-cloud-auto
calibrate-cloud-auto: ## Run cloud calibration with auto-size/progress. Pass CALIBRATE_ARGS='...'.
	$(UV) run --group cloudops $(PYTHON) -m example_model.calibrate --cloud --cloud-config $(CLOUD_CONFIG) --auto-size --print-task-progress $(CALIBRATE_ARGS)

.PHONY: cloud-cleanup-preview
cloud-cleanup-preview: ## Preview cleanup for SESSION_ID=... Optional: IMAGE_TAG=...
	$(MAKE) cloud-cleanup SESSION_ID="$(SESSION_ID)" IMAGE_TAG="$(IMAGE_TAG)" SKIP_ACR="$(SKIP_ACR)" DRY_RUN=1

.PHONY: cloud-cleanup-session
cloud-cleanup-session: ## Delete cloud resources for SESSION_ID=... Optional: IMAGE_TAG=...
	$(MAKE) cloud-cleanup SESSION_ID="$(SESSION_ID)" IMAGE_TAG="$(IMAGE_TAG)" SKIP_ACR="$(SKIP_ACR)" DRY_RUN=

.PHONY: cloud-cleanup-user-preview
cloud-cleanup-user-preview: ## Preview cleanup for CLOUD_USER; defaults to current user.
	$(MAKE) cloud-cleanup-user CLOUD_USER="$(CLOUD_USER)" IMAGE_TAG="$(IMAGE_TAG)" SKIP_ACR="$(SKIP_ACR)" DRY_RUN=1

.PHONY: cloud-cleanup-user-delete
cloud-cleanup-user-delete: ## Delete all cloud sessions for CLOUD_USER; defaults to current user.
	$(MAKE) cloud-cleanup-user CLOUD_USER="$(CLOUD_USER)" IMAGE_TAG="$(IMAGE_TAG)" SKIP_ACR="$(SKIP_ACR)" DRY_RUN=

.PHONY: cloud-list
cloud-list: ## List cloud resources. Optional: SESSION_ID=... CLOUD_USER=... IMAGE_TAG=... SKIP_ACR=1
	$(UV) run --group cloudops $(PYTHON) -m calibrationtools.cloud.cleanup --cloud-config $(CLOUD_CONFIG) --list $(SESSION_ID_FLAG) $(USER_FLAG) $(IMAGE_TAG_FLAG) $(SKIP_ACR_FLAG)

.PHONY: cloud-cleanup-plan
cloud-cleanup-plan: cloud-cleanup-preview

.PHONY: cloud-cleanup-delete
cloud-cleanup-delete: cloud-cleanup-session

.PHONY: cloud-cleanup
cloud-cleanup:
	@test -n "$(SESSION_ID)" || (echo "SESSION_ID is required"; exit 1)
	$(UV) run --group cloudops $(PYTHON) -m calibrationtools.cloud.cleanup --cloud-config $(CLOUD_CONFIG) $(SESSION_ID_FLAG) $(IMAGE_TAG_FLAG) $(SKIP_ACR_FLAG) $(DRY_RUN_FLAG)

.PHONY: cloud-cleanup-user
cloud-cleanup-user:
	@test -n "$(CLOUD_USER)" || (echo "CLOUD_USER is required"; exit 1)
	$(UV) run --group cloudops $(PYTHON) -m calibrationtools.cloud.cleanup --cloud-config $(CLOUD_CONFIG) --all-sessions-for-user $(USER_FLAG) $(IMAGE_TAG_FLAG) $(SKIP_ACR_FLAG) $(DRY_RUN_FLAG)

.PHONY: help-vars
help-vars:
	@printf '%s\n' \
		'Common variables:' \
		"  TEST_ARGS='-k sampler'                 Filter pytest" \
		"  MRP_ARGS='--input seed=42'             Pass inputs to mrp run" \
		"  CALIBRATE_ARGS='--artifacts-dir /tmp/run'" \
		'                                         Pass options to example calibration' \
		"  RUFF_ARGS='--fix'                      Pass extra ruff options" \
		"  TY_ARGS='--ignore=unresolved-import'   Pass extra ty options" \
		'  CLOUD_CONFIG=path/to/cloud.toml        Use another cloud config' \
		'  SESSION_ID=...                         Select one cloud session' \
		'  CLOUD_USER=...                         Select cloud sessions for one user' \
		'  IMAGE_TAG=...                          Filter cloud image resources' \
		'  SKIP_ACR=1                             Skip Azure Container Registry lookup/cleanup' \
		'  DRY_RUN=1                              Preview cleanup without deletion' \
		'' \
		'Tool and path overrides:' \
		'  UV=uv                                  uv executable' \
		'  PYTHON=python                          Python executable used by uv' \
		'  PRE_COMMIT=pre-commit                  pre-commit executable' \
		'  DOCKER=docker                          Docker executable' \
		'  EXAMPLE_PACKAGE=example-model          Example model package name' \
		'  DOCKER_IMAGE=...                       Example model Docker image tag' \
		'  DOCKERFILE=packages/example_model/Dockerfile' \
		'                                         Example model Dockerfile path' \
		'  MRP_CONFIG=path/to/config.toml         Inline MRP config' \
		'  MRP_DOCKER_CONFIG=path/to/config.toml  Docker-backed MRP config' \
		'' \
		'Examples:' \
		"  make test TEST_ARGS='-k sampler'" \
		"  make lint RUFF_ARGS='--fix'" \
		"  make typecheck TY_ARGS='--ignore=unresolved-import'" \
		"  make mrp MRP_ARGS='--input seed=42 --input max_gen=10'" \
		"  make calibrate CALIBRATE_ARGS='--artifacts-dir /tmp/run'" \
		'  make cloud-cleanup-preview SESSION_ID=20260412010101-alice-testsha-ab12cd34ef56'

.PHONY: help-cloud
help-cloud:
	@printf '%s\n' \
		'Cloud workflow:' \
		'  make setup-cloud' \
		'  make cloud-run-auto' \
		'  make cloud-list' \
		'  make cloud-cleanup-preview SESSION_ID=...' \
		'  make cloud-cleanup-session SESSION_ID=...' \
		'  make cloud-cleanup-user-preview' \
		'  make cloud-cleanup-user-delete' \
		'' \
		'Cleanup safety:' \
		'  Preview targets set DRY_RUN=1 and do not delete resources.' \
		'  cloud-cleanup-session requires SESSION_ID and deletes that session after printing the plan.' \
		'  CLOUD_USER defaults to the current shell user for user-wide list and cleanup commands.' \
		'  Pass CLOUD_USER=other-user only when you intend to inspect or clean that user.' \
		'  IMAGE_TAG=... narrows cleanup to matching cloud image resources.' \
		'  SKIP_ACR=1 skips Azure Container Registry lookup and image tag cleanup.' \
		'' \
		'Examples:' \
		'  make cloud-run-auto' \
		'  make cloud-list' \
		'  make cloud-list SKIP_ACR=1' \
		'  make cloud-list CLOUD_USER=alice' \
		'  make cloud-cleanup-preview SESSION_ID=20260412010101-alice-testsha-ab12cd34ef56' \
		'  make cloud-cleanup-session SESSION_ID=20260412010101-alice-testsha-ab12cd34ef56' \
		'  make cloud-cleanup-user-preview' \
		'  make cloud-cleanup-user-delete CLOUD_USER=alice' \
		'' \
		'Compatibility aliases:' \
		'  cloud-cleanup-plan        -> cloud-cleanup-preview' \
		'  cloud-cleanup-delete      -> cloud-cleanup-session' \
		'  cloud-cleanup             -> underlying SESSION_ID cleanup target' \
		'  cloud-cleanup-user        -> underlying CLOUD_USER cleanup target'
