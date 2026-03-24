#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import shutil
import sys

from azure.core.exceptions import HttpResponseError, ResourceNotFoundError

def parse_image_ref(image_ref: str) -> tuple[str, str]:
    if ":" not in image_ref:
        raise SystemExit(
            "Image must be provided as `repository:tag`, for example "
            "`cfa-calibration-tools-example-model:latest`."
        )
    repository, tag = image_ref.rsplit(":", 1)
    if not repository or not tag:
        raise SystemExit(
            "Image must be provided as `repository:tag`, for example "
            "`cfa-calibration-tools-example-model:latest`."
        )
    return repository, tag


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check whether selected CloudOps resources exist."
    )
    parser.add_argument(
        "--image",
        help="Image to check in ACR, formatted as `repository:tag`.",
    )
    parser.add_argument(
        "--pool",
        help="Azure Batch pool name to check.",
    )
    parser.add_argument(
        "--job",
        help="Azure Batch job name to check.",
    )
    args = parser.parse_args()
    if not any([args.image, args.pool, args.job]):
        parser.error("Provide at least one of `--image`, `--pool`, or `--job`.")
    return args


def print_status(label: str, detail: str) -> None:
    print(f"{label}: {detail}")


def require_tool(name: str) -> None:
    if shutil.which(name) is None:
        if name == "az":
            raise SystemExit(
                "Azure CLI (`az`) is required for image checks because CloudOps uses "
                "it to query ACR tags.\n"
                "Install Azure CLI and make sure `az --version` works in this shell.\n"
                "In Powershell: winget install --exact --id Microsoft.AzureCLI\n"
                "If Azure CLI is already installed, restart the shell so PATH updates "
                "take effect."
            )
        raise SystemExit(f"Required tool not found on PATH: {name}")


def main() -> int:
    print_status("Main status", "starting check_runner")
    args = parse_args()
    print_status("Main status", "parsed arguments")
    try:
        from cfa.cloudops import CloudClient, batch_helpers
    except ImportError as exc:
        raise SystemExit(
            "Could not import cfa.cloudops. Run `uv sync --all-packages --group cloudops` "
            "or `uv run --group cloudops python packages/example_model/check_runner.py`."
        ) from exc
    print_status("Client status", "initializing CloudClient")
    client = CloudClient(keyvault="cfa-predict")
    print_status("Client status", "CloudClient initialized")

    summary: dict[str, object] = {}

    if args.image:
        require_tool("az")
        registry_account = client.cred.azure_container_registry_account
        if not registry_account:
            raise SystemExit(
                "AZURE_CONTAINER_REGISTRY_ACCOUNT must be set in the environment or .env."
            )
        repository, tag = parse_image_ref(args.image)
        print_status("Image status", f"checking {repository}:{tag} in {registry_account}")
        try:
            image_exists = tag in client.list_acr_tags(
                registry_name=registry_account,
                repo_name=repository,
            )
        except Exception as exc:
            raise SystemExit(
                f"Failed to query ACR tags for {repository} with CloudOps/Azure CLI: {exc}"
            ) from exc
        print_status("Image status", "found" if image_exists else "not found")
        summary["image"] = {
            "registry": registry_account,
            "repository": repository,
            "tag": tag,
            "exists": image_exists,
        }

    if args.pool:
        if not client.cred.azure_resource_group_name:
            raise SystemExit(
                "AZURE_RESOURCE_GROUP_NAME must be set in the environment or .env."
            )
        if not client.cred.azure_batch_account:
            raise SystemExit(
                "AZURE_BATCH_ACCOUNT must be set in the environment or .env."
            )
        print_status("Pool status", f"checking {args.pool}")
        pool_exists = batch_helpers.check_pool_exists(
            resource_group_name=client.cred.azure_resource_group_name,
            account_name=client.cred.azure_batch_account,
            pool_name=args.pool,
            batch_mgmt_client=client.batch_mgmt_client,
        )
        print_status("Pool status", "found" if pool_exists else "not found")
        summary["pool"] = {
            "name": args.pool,
            "exists": pool_exists,
        }

    if args.job or args.pool:
        if args.job:
            print_status("Job status", "loading Batch jobs")
        else:
            print_status("Jobs status", f"loading jobs for pool {args.pool}")
        try:
            jobs = list(client.batch_service_client.job.list())
        except HttpResponseError as exc:
            raise SystemExit(f"Failed to list Azure Batch jobs: {exc}") from exc

    if args.job:
        print_status("Job status", f"checking {args.job}")
        specific_job_exists = batch_helpers.check_job_exists(
            args.job, client.batch_service_client
        )
        specific_job_state = None
        specific_job_pool = None
        if specific_job_exists:
            try:
                job = client.batch_service_client.job.get(args.job)
                specific_job_state = getattr(job.state, "value", job.state)
                if getattr(job, "pool_info", None) is not None:
                    specific_job_pool = getattr(job.pool_info, "pool_id", None)
            except ResourceNotFoundError:
                specific_job_exists = False
        print_status("Job status", "found" if specific_job_exists else "not found")
        summary["job"] = {
            "name": args.job,
            "exists": specific_job_exists,
            "state": specific_job_state,
            "pool": specific_job_pool,
        }

    if args.pool and not args.job:
        jobs_in_pool = []
        for job in jobs:
            pool_info = getattr(job, "pool_info", None)
            if pool_info is None:
                continue
            if getattr(pool_info, "pool_id", None) == args.pool:
                jobs_in_pool.append(job.id)
        print_status(
            "Jobs status",
            f"{len(jobs_in_pool)} job(s) found for pool {args.pool}",
        )
        summary["job"] = {
            "pool": args.pool,
            "count": len(jobs_in_pool),
            "exists": len(jobs_in_pool) > 0,
            "ids": jobs_in_pool,
        }

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
