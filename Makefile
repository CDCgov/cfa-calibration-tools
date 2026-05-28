.DEFAULT_GOAL := help

##@var Tool and path overrides||uv executable
UV ?= uv
##@var Tool and path overrides||Python executable used by uv
PYTHON ?= python
##@var Tool and path overrides||pre-commit executable
PRE_COMMIT ?= pre-commit
##@var Tool and path overrides||Docker executable
DOCKER ?= docker
RUFF_VERSION ?= 0.14.6

##@var Tool and path overrides||Example model package name
EXAMPLE_PACKAGE ?= example-model
##@var Tool and path overrides|...|Example model Docker image tag
DOCKER_IMAGE ?= cfa-calibration-tools-example-model-python:latest
##@var Tool and path overrides||Example model Dockerfile path
DOCKERFILE ?= packages/example_model/Dockerfile

##@var Tool and path overrides|path/to/config.toml|Inline MRP config
MRP_CONFIG ?= packages/example_model/src/example_model/example_model.mrp.toml
##@var Tool and path overrides|path/to/config.toml|Docker-backed MRP config
MRP_DOCKER_CONFIG ?= packages/example_model/src/example_model/example_model.mrp.docker.toml
##@var Common variables|path/to/cloud.toml|Use another cloud config
CLOUD_CONFIG ?= packages/example_model/src/example_model/example_model.cloud_config.toml

# Passthrough variables append raw CLI arguments after target-owned defaults.
# Example: make calibrate-cloud CALIBRATE_ARGS='--artifacts-dir /tmp/run'
##@var Common variables|'--input seed=42'|Pass inputs to mrp run
MRP_ARGS ?=
##@var Common variables|'--artifacts-dir /tmp/run'|Pass options to example calibration
CALIBRATE_ARGS ?=
##@var Common variables|3|Enable per-slot speculative lookahead for calibration
SLOT_LOOKAHEAD ?=
##@var Common variables|'-k sampler'|Filter pytest
TEST_ARGS ?=
##@var Common variables|'--fix'|Pass extra ruff options
RUFF_ARGS ?=
##@var Common variables|'--ignore=unresolved-import'|Pass extra ty options
TY_ARGS ?=

##@var Common variables|...|Select one cloud session
SESSION_ID ?=
##@var Common variables|...|Select cloud sessions for one user
CLOUD_USER ?=
DEFAULT_CLOUD_USER ?= $(or $(shell python3 -c 'import getpass; print(getpass.getuser())' 2>/dev/null | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9-]+/-/g; s/^-+//; s/-+$$//; s/-+/-/g'),$(firstword $(shell id -un 2>/dev/null || whoami 2>/dev/null || printf unknown)))
##@var Common variables|1|Preview cleanup without deletion
DRY_RUN ?=
##@var Common variables|...|Filter cloud image resources
IMAGE_TAG ?=
##@var Common variables|1|Skip Azure Container Registry lookup/cleanup
SKIP_ACR ?=
CLEANUP_USER = $(or $(strip $(CLOUD_USER)),$(DEFAULT_CLOUD_USER))

ifneq ($(strip $(SESSION_ID)),)
SESSION_ID_FLAG := --session-id $(SESSION_ID)
endif

ifneq ($(strip $(CLOUD_USER)),)
USER_FLAG := --user $(CLOUD_USER)
endif

ifneq ($(strip $(CLEANUP_USER)),)
CLEANUP_USER_FLAG := --user $(CLEANUP_USER)
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

ifneq ($(strip $(SLOT_LOOKAHEAD)),)
SLOT_LOOKAHEAD_FLAG := --slot-lookahead $(SLOT_LOOKAHEAD)
endif

PYTEST_CMD = $(UV) run pytest
RUFF_CMD = $(UV) run --with ruff==$(RUFF_VERSION) ruff
RUFF_COMMON_ARGS = --line-length 79 .
TY_CMD = $(UV) run ty
MRP_CMD = $(UV) run --package $(EXAMPLE_PACKAGE) mrp run
CALIBRATE_CMD = $(UV) run $(PYTHON) -m example_model.calibrate
CLOUD_CALIBRATE_CMD = $(UV) run --group cloudops $(PYTHON) -m example_model.calibrate --cloud --cloud-config $(CLOUD_CONFIG)
CALIBRATE_COMMON_ARGS = $(SLOT_LOOKAHEAD_FLAG) $(CALIBRATE_ARGS)
CLOUD_CLEANUP_CMD = $(UV) run --group cloudops $(PYTHON) -m calibrationtools.cloud.cleanup --cloud-config $(CLOUD_CONFIG)
CLOUD_CLEANUP_SESSION_VARS = SESSION_ID="$(SESSION_ID)" IMAGE_TAG="$(IMAGE_TAG)" SKIP_ACR="$(SKIP_ACR)"
CLOUD_CLEANUP_USER_VARS = CLOUD_USER="$(CLEANUP_USER)" IMAGE_TAG="$(IMAGE_TAG)" SKIP_ACR="$(SKIP_ACR)"

##@example Variables|make test TEST_ARGS='-k sampler'
##@example Variables|make lint RUFF_ARGS='--fix'
##@example Variables|make typecheck TY_ARGS='--ignore=unresolved-import'
##@example Variables|make mrp MRP_ARGS='--input seed=42 --input max_gen=10'
##@example Variables|make calibrate CALIBRATE_ARGS='--artifacts-dir /tmp/run'
##@example Variables|make calibrate-docker SLOT_LOOKAHEAD=3
##@example Variables|make calibrate-cloud SLOT_LOOKAHEAD=3
##@example Variables|make cloud-cleanup-preview SESSION_ID=20260412010101-alice-testsha-ab12cd34ef56

# Alias targets stay out of the main help so each operation has one canonical
# listing while old and convenience target names keep working.
.PHONY: setup ci example example-mrp example-docker example-benchmark
.PHONY: setup-cloud cloud-run cloud-run-auto
.PHONY: cloud-cleanup-plan cloud-cleanup-delete
setup: sync ##@alias General
ci: check ##@alias General

example: calibrate ##@alias Example
example-mrp: mrp ##@alias Example
example-docker: calibrate-docker ##@alias Example
example-benchmark: benchmark ##@alias Example

setup-cloud: sync-cloud ##@alias Cloud
cloud-run: calibrate-cloud ##@alias Cloud
cloud-run-auto: calibrate-cloud-auto ##@alias Cloud
cloud-cleanup-plan: cloud-cleanup-preview ##@alias Cloud
cloud-cleanup-delete: cloud-cleanup-session ##@alias Cloud

##@ Setup

.PHONY: sync
sync: ## Sync all workspace packages and extras.
	$(UV) sync --all-packages --all-extras

.PHONY: sync-cloud
sync-cloud: ## Sync workspace packages with CloudOps dependencies.
	$(UV) sync --all-packages --group cloudops

.PHONY: lock
lock: ## Update uv.lock.
	$(UV) lock

##@ Quality

.PHONY: test
test: ## Run the full pytest suite. Pass TEST_ARGS='...' for filters.
	$(PYTEST_CMD) $(TEST_ARGS)

.PHONY: fix
fix: ## Run all pre-commit hooks, applying autofixes, then verify once.
	$(PRE_COMMIT) run --all-files || $(PRE_COMMIT) run --all-files

.PHONY: test-core
test-core: ## Run tests for the calibrationtools package.
	$(PYTEST_CMD) tests $(TEST_ARGS)

.PHONY: test-example
test-example: ## Run tests for the bundled example model package.
	$(PYTEST_CMD) packages/example_model/tests $(TEST_ARGS)

.PHONY: lint
lint: ## Run ruff checks.
	$(RUFF_CMD) check $(RUFF_COMMON_ARGS) $(RUFF_ARGS)

.PHONY: format
format: ## Format Python code with ruff.
	$(RUFF_CMD) format $(RUFF_COMMON_ARGS) $(RUFF_ARGS)

.PHONY: format-check
format-check: ## Check Python formatting.
	$(RUFF_CMD) format --check $(RUFF_COMMON_ARGS) $(RUFF_ARGS)

.PHONY: typecheck
typecheck: ## Run ty type checks.
	$(TY_CMD) check --ignore=unresolved-import $(TY_ARGS)

.PHONY: precommit
precommit: ## Run all configured pre-commit hooks.
	$(PRE_COMMIT) run --all-files

.PHONY: check
check: precommit test ## Run precommit, and tests.

##@ Example model

.PHONY: docker-build
docker-build: ## Build the example model Docker image.
	$(DOCKER) build -t $(DOCKER_IMAGE) -f $(DOCKERFILE) .

.PHONY: mrp
mrp: ## Run the default inline MRP config. Pass MRP_ARGS='--input seed=42'.
	$(MRP_CMD) $(MRP_CONFIG) $(MRP_ARGS)

.PHONY: mrp-docker
mrp-docker: docker-build ## Build the image and run the Docker-backed MRP config.
	$(MRP_CMD) $(MRP_DOCKER_CONFIG) $(MRP_ARGS)

.PHONY: calibrate
calibrate: ## Run local calibration. Pass CALIBRATE_ARGS='...'.
	$(CALIBRATE_CMD) $(CALIBRATE_COMMON_ARGS)

.PHONY: calibrate-docker
calibrate-docker: docker-build ## Run Docker-backed calibration. Pass CALIBRATE_ARGS='...'.
	$(CALIBRATE_CMD) --docker $(CALIBRATE_COMMON_ARGS)

.PHONY: benchmark
benchmark: ## Compare serial and parallel example calibration execution.
	$(UV) run $(PYTHON) -m example_model.benchmark

##@ Cloud

.PHONY: calibrate-cloud
calibrate-cloud: ## Run cloud-backed calibration. Pass CALIBRATE_ARGS='...'.
	$(CLOUD_CALIBRATE_CMD) $(CALIBRATE_COMMON_ARGS)

.PHONY: calibrate-cloud-auto
calibrate-cloud-auto: ## Run cloud calibration with auto-size/progress. Pass CALIBRATE_ARGS='...'.
	$(CLOUD_CALIBRATE_CMD) --auto-size --print-task-progress $(CALIBRATE_COMMON_ARGS)

.PHONY: cloud-cleanup-preview
cloud-cleanup-preview: ## Preview cleanup for SESSION_ID=... Optional: IMAGE_TAG=...
	$(MAKE) cloud-cleanup $(CLOUD_CLEANUP_SESSION_VARS) DRY_RUN=1

.PHONY: cloud-cleanup-session
cloud-cleanup-session: ## Delete cloud resources for SESSION_ID=... Optional: IMAGE_TAG=...
	$(MAKE) cloud-cleanup $(CLOUD_CLEANUP_SESSION_VARS) DRY_RUN=

.PHONY: cloud-cleanup-user-preview
cloud-cleanup-user-preview: ## Preview cleanup for CLOUD_USER; defaults to current user.
	$(MAKE) cloud-cleanup-user $(CLOUD_CLEANUP_USER_VARS) DRY_RUN=1

.PHONY: cloud-cleanup-user-delete
cloud-cleanup-user-delete: ## Delete all cloud sessions for CLOUD_USER; defaults to current user.
	$(MAKE) cloud-cleanup-user $(CLOUD_CLEANUP_USER_VARS) DRY_RUN=

.PHONY: cloud-list
cloud-list: ## List cloud resources. Optional: SESSION_ID=... CLOUD_USER=... IMAGE_TAG=... SKIP_ACR=1
	$(CLOUD_CLEANUP_CMD) --list $(SESSION_ID_FLAG) $(USER_FLAG) $(IMAGE_TAG_FLAG) $(SKIP_ACR_FLAG)

.PHONY: cloud-cleanup
cloud-cleanup: ##@cloud-helper underlying SESSION_ID cleanup target
	@test -n "$(SESSION_ID)" || (echo "SESSION_ID is required"; exit 1)
	$(CLOUD_CLEANUP_CMD) $(SESSION_ID_FLAG) $(IMAGE_TAG_FLAG) $(SKIP_ACR_FLAG) $(DRY_RUN_FLAG)

.PHONY: cloud-cleanup-user
cloud-cleanup-user: ##@cloud-helper underlying CLOUD_USER cleanup target
	@test -n "$(CLEANUP_USER)" || (echo "CLOUD_USER is required"; exit 1)
	$(CLOUD_CLEANUP_CMD) --all-sessions-for-user $(CLEANUP_USER_FLAG) $(IMAGE_TAG_FLAG) $(SKIP_ACR_FLAG) $(DRY_RUN_FLAG)

##@cloud-note Preview targets set DRY_RUN=1 and do not delete resources.
##@cloud-note cloud-cleanup-session requires SESSION_ID and deletes that session after printing the plan.
##@cloud-note CLOUD_USER filters cloud-list and selects user cleanup sessions.
##@cloud-note cloud-cleanup-user-* defaults to the current shell user when CLOUD_USER is unset.
##@cloud-note Pass CLOUD_USER=other-user only when you intend to inspect or clean that user.
##@cloud-note IMAGE_TAG=... narrows cleanup to matching cloud image resources.
##@cloud-note SKIP_ACR=1 skips Azure Container Registry lookup and image tag cleanup.

##@example Cloud|make cloud-run-auto
##@example Cloud|make cloud-list
##@example Cloud|make cloud-list SKIP_ACR=1
##@example Cloud|make cloud-list CLOUD_USER=alice
##@example Cloud|make cloud-cleanup-preview SESSION_ID=20260412010101-alice-testsha-ab12cd34ef56
##@example Cloud|make cloud-cleanup-session SESSION_ID=20260412010101-alice-testsha-ab12cd34ef56
##@example Cloud|make cloud-cleanup-user-preview
##@example Cloud|make cloud-cleanup-user-delete CLOUD_USER=alice

##@ Help

.PHONY: help
help: ## Show available targets.
	@awk 'BEGIN {FS = ":.*## "; printf "Usage:\n  make <target> [VAR=value]\n"} /^##@ / {printf "\n%s:\n", substr($$0, 5); next} /^[a-zA-Z0-9_.-]+:.*## / {printf "  %-28s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

.PHONY: help-vars
help-vars: ## Show documented Makefile variables and usage examples.
	@awk 'BEGIN {pending = 0} /^##@var / {payload = $$0; sub(/^##@var[ \t]*/, "", payload); split(payload, parts, /\|/); group = parts[1]; sample = parts[2]; desc = parts[3]; pending = 1; next} pending && /^[A-Za-z_][A-Za-z0-9_]*[ \t]*[?:+!]?=/ {line = $$0; var = line; sub(/[ \t]*[?:+!]?=.*/, "", var); value = line; sub(/^[^=]*=[ \t]*/, "", value); sub(/[ \t]*#.*/, "", value); if (group != current) {if (current != "") print ""; print group ":"; current = group} display = sample != "" ? sample : value; if (display != "") display = var "=" display; else display = var; printf "  %-50s %s\n", display, desc; pending = 0; next} /^##@example Variables\|/ {line = $$0; sub(/^##@example Variables\|/, "", line); examples[++example_count] = line} END {if (example_count) {print "\nExamples:"; for (i = 1; i <= example_count; i++) print "  " examples[i]}}' $(MAKEFILE_LIST)

.PHONY: help-aliases
help-aliases: ## Show compatibility and convenience aliases.
	@awk '/^[a-zA-Z0-9_.-]+:[^#]*##@alias/ {line = $$0; target = line; sub(/:.*/, "", target); deps = line; sub(/^[^:]+:[ \t]*/, "", deps); sub(/[ \t]*##@alias.*/, "", deps); split(deps, dep_parts, /[ \t]+/); canonical = dep_parts[1]; group = line; sub(/^.*##@alias[ \t]*/, "", group); if (group == "" || group == line) group = "Aliases"; sub(/[ \t].*/, "", group); if (group != current) {if (current != "") print ""; print group ":"; current = group} printf "  %-24s -> %s\n", target, canonical}' $(MAKEFILE_LIST)

.PHONY: help-cloud
help-cloud: ## Show cloud targets, aliases, safety notes, and examples.
	@awk 'BEGIN {section = ""} /^##@ / {section = substr($$0, 5); next} section == "Cloud" && /^[a-zA-Z0-9_.-]+:.*## / {split($$0, parts, ":.*## "); targets[++target_count] = sprintf("  %-28s %s", parts[1], parts[2]); next} /^[a-zA-Z0-9_.-]+:[^#]*##@alias[ \t]+Cloud/ {line = $$0; target = line; sub(/:.*/, "", target); deps = line; sub(/^[^:]+:[ \t]*/, "", deps); sub(/[ \t]*##@alias.*/, "", deps); split(deps, dep_parts, /[ \t]+/); aliases[++alias_count] = sprintf("  %-28s -> %s", target, dep_parts[1]); next} /^[a-zA-Z0-9_.-]+:[^#]*##@cloud-helper / {line = $$0; target = line; sub(/:.*/, "", target); desc = line; sub(/^.*##@cloud-helper[ \t]*/, "", desc); helpers[++helper_count] = sprintf("  %-28s -> %s", target, desc); next} /^##@cloud-note / {line = $$0; sub(/^##@cloud-note[ \t]*/, "", line); notes[++note_count] = line; next} /^##@example Cloud\|/ {line = $$0; sub(/^##@example Cloud\|/, "", line); examples[++example_count] = line; next} END {print "Cloud targets:"; for (i = 1; i <= target_count; i++) print targets[i]; if (alias_count) {print "\nCloud aliases:"; for (i = 1; i <= alias_count; i++) print aliases[i]} if (helper_count) {print "\nUnderlying cleanup targets:"; for (i = 1; i <= helper_count; i++) print helpers[i]} if (note_count) {print "\nCleanup safety:"; for (i = 1; i <= note_count; i++) print "  " notes[i]} if (example_count) {print "\nExamples:"; for (i = 1; i <= example_count; i++) print "  " examples[i]}}' $(MAKEFILE_LIST)
