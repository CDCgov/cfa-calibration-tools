#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


"""
This script needs the following in action secrets define in environment variables or in a .env file.
AZURE_BATCH_ACCOUNT=
AZURE_USER_ASSIGNED_IDENTITY=
AZURE_SUBNET_ID=
AZURE_CLIENT_ID=
AZURE_KEYVAULT_NAME=
AZURE_KEYVAULT_SP_SECRET_ID=

# Azure Blob storage config
AZURE_BLOB_STORAGE_ACCOUNT=

# Azure container registry config
AZURE_CONTAINER_REGISTRY_ACCOUNT=

# Azure SP info
AZURE_TENANT_ID=
AZURE_SUBSCRIPTION_ID=
"""

DEFAULT_LOCAL_IMAGE = "cfa-calibration-tools-example-model-python"
DEFAULT_REMOTE_REPOSITORY = "cfa-calibration-tools-example-model"


def require_tool(name: str) -> None:
    if shutil.which(name) is None:
        raise SystemExit(f"Required tool not found on PATH: {name}")


def run_command(command: list[str], *, cwd: Path) -> None:
    print(f"+ {' '.join(command)}")
    subprocess.run(command, cwd=cwd, check=True)


def git_short_sha(repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the example model runner image locally and publish it with CloudOps."
    )
    parser.add_argument(
        "--repository",
        default=DEFAULT_REMOTE_REPOSITORY,
        help=f"Remote image repository name. Defaults to {DEFAULT_REMOTE_REPOSITORY}.",
    )
    parser.add_argument(
        "--local-image",
        default=DEFAULT_LOCAL_IMAGE,
        help=f"Local image name to build. Defaults to {DEFAULT_LOCAL_IMAGE}.",
    )
    parser.add_argument(
        "--tag",
        help="Image tag to publish. Defaults to the current git short SHA.",
    )
    parser.add_argument(
        "--push-latest",
        action="store_true",
        help="Also tag and push the image as latest.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    require_tool("docker")
    require_tool("git")
    require_tool("az")

    repo_root = Path(__file__).resolve().parents[2]
    dockerfile = repo_root / "packages" / "example_model" / "Dockerfile"

    tag = args.tag or git_short_sha(repo_root)
    local_image_ref = f"{args.local_image}:{tag}"

    run_command(
        ["docker", "build", "-t", local_image_ref, "-f", str(dockerfile), "."],
        cwd=repo_root,
    )

    try:
        from cfa.cloudops import CloudClient
    except ImportError as exc:
        raise SystemExit(
            "Could not import cfa.cloudops. Run `uv sync --all-packages --group cloudops` "
            "or `uv run --group cloudops python packages/example_model/create_runner.py`."
        ) from exc

    # For local runs we want DefaultAzureCredential behavior, not managed identity IMDS.
    client = CloudClient(use_federated=True)
    registry_account = client.cred.azure_container_registry_account

    if not registry_account:
        raise SystemExit(
            "AZURE_CONTAINER_REGISTRY_ACCOUNT must be set in the environment or .env."
        )

    # CloudOps currently shells out to Azure CLI for image uploads. Outside Azure-hosted
    # managed identity environments, its non-device-code path uses `az login --identity`,
    # which fails locally.
    use_device_code = os.environ.get("GITHUB_ACTIONS", "").lower() != "true"

    remote_image_ref = client.upload_docker_image(
        image_name=local_image_ref,
        registry_name=registry_account,
        repo_name=args.repository,
        tag=tag,
        use_device_code=use_device_code,
    )

    if args.push_latest:
        latest_remote_ref = client.upload_docker_image(
            image_name=local_image_ref,
            registry_name=registry_account,
            repo_name=args.repository,
            tag="latest",
            use_device_code=use_device_code,
        )
        print(f"Published {remote_image_ref} and {latest_remote_ref}")
    else:
        print(f"Published {remote_image_ref}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
