"""Container registry resolution and worker image publication."""

from __future__ import annotations

import os
from typing import Any


def resolve_registry_server(
    explicit: str | None, cloud_client: Any | None
) -> str:
    """Resolve an ACR server from explicit config or cfa-cloudops settings."""

    registry = explicit or os.getenv("AZURE_CONTAINER_REGISTRY_SERVER")
    if registry:
        return registry
    if cloud_client is not None:
        try:
            registry = cloud_client.cred.azure_container_registry_endpoint
        except AttributeError:
            registry = None
        if registry:
            return registry
    account = os.getenv("AZURE_CONTAINER_REGISTRY_ACCOUNT")
    if account:
        domain = os.getenv("AZURE_CONTAINER_REGISTRY_DOMAIN", "azurecr.io")
        return f"{account}.{domain}"
    raise ValueError(
        "Container registry server must be configured through "
        "registry_server, AZURE_CONTAINER_REGISTRY_SERVER, or "
        "AZURE_CONTAINER_REGISTRY_ACCOUNT"
    )


def build_and_push_image(
    cloud_client: Any,
    *,
    registry: str,
    image_name: str,
    image_tag: str,
    image_dockerfile: str,
) -> None:
    """Build the worker image from a Dockerfile and push it to the registry."""

    registry_name = registry.removesuffix(".azurecr.io")
    cloud_client.package_and_upload_dockerfile(
        registry_name=registry_name,
        repo_name=image_name,
        tag=image_tag,
        path_to_dockerfile=image_dockerfile,
        use_device_code=False,
    )
    try:
        tags = cloud_client.list_acr_tags(
            registry_name=registry_name, repo_name=image_name
        )
    except Exception as exc:
        raise RuntimeError(
            "Azure image publication could not be verified. Authenticate "
            "the container runtime to ACR and push the image before "
            "retrying."
        ) from exc
    if image_tag not in tags:
        raise RuntimeError(
            "Azure image publication did not produce "
            f"{image_name}:{image_tag} in {registry}."
        )


def upload_image(
    cloud_client: Any,
    *,
    registry: str,
    image_name: str,
    image_tag: str,
) -> None:
    """Push an already-built local image to the registry."""

    cloud_client.upload_docker_image(
        image_name=f"{image_name}:{image_tag}",
        registry_name=registry.removesuffix(".azurecr.io"),
        repo_name=image_name,
        tag=image_tag,
        use_device_code=False,
    )
