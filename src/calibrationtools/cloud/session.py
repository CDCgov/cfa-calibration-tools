from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..run_id import parse_sampler_run_id
from .naming import (
    format_generation_name,
    parse_generation_from_run_id,
)


def _normalize_job_name_list(value: str | list[str]) -> list[str]:
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


@dataclass(frozen=True)
class CloudSession:
    keyvault: str
    session_id: str
    image_tag: str
    remote_image_ref: str
    pool_name: str
    job_names: dict[str, list[str]]
    input_container: str
    output_container: str
    logs_container: str
    task_mrp_config_path: str
    input_mount_path: str
    output_mount_path: str
    logs_mount_path: str
    task_timeout_minutes: int | None
    print_task_durations: bool

    def to_runtime_cloud(self) -> dict[str, Any]:
        return {
            "keyvault": self.keyvault,
            "session_id": self.session_id,
            "image_tag": self.image_tag,
            "remote_image_ref": self.remote_image_ref,
            "pool_name": self.pool_name,
            "job_names": {
                key: list(value) for key, value in self.job_names.items()
            },
            "input_container": self.input_container,
            "output_container": self.output_container,
            "logs_container": self.logs_container,
            "task_mrp_config_path": self.task_mrp_config_path,
            "input_mount_path": self.input_mount_path,
            "output_mount_path": self.output_mount_path,
            "logs_mount_path": self.logs_mount_path,
            "task_timeout_minutes": self.task_timeout_minutes,
            "print_task_durations": self.print_task_durations,
        }

    @classmethod
    def from_runtime_cloud(cls, cloud: dict[str, Any]) -> "CloudSession":
        return cls(
            keyvault=cloud["keyvault"],
            session_id=cloud["session_id"],
            image_tag=cloud["image_tag"],
            remote_image_ref=cloud["remote_image_ref"],
            pool_name=cloud["pool_name"],
            job_names={
                str(key): _normalize_job_name_list(value)
                for key, value in cloud["job_names"].items()
            },
            input_container=cloud["input_container"],
            output_container=cloud["output_container"],
            logs_container=cloud["logs_container"],
            task_mrp_config_path=cloud["task_mrp_config_path"],
            input_mount_path=cloud["input_mount_path"],
            output_mount_path=cloud["output_mount_path"],
            logs_mount_path=cloud["logs_mount_path"],
            task_timeout_minutes=cloud.get("task_timeout_minutes"),
            print_task_durations=bool(
                cloud.get("print_task_durations", False)
            ),
        )

    def job_name_for_run(self, run_id: str) -> str:
        parsed_run_id = parse_sampler_run_id(run_id)
        generation = str(parsed_run_id.generation_index)
        try:
            job_names = _normalize_job_name_list(self.job_names[generation])
        except KeyError as exc:
            raise KeyError(
                f"No Azure Batch job configured for generation {generation}"
            ) from exc
        return job_names[parsed_run_id.proposal_index % len(job_names)]

    def logs_folder_for_job(
        self, job_name: str, run_id: str | None = None
    ) -> str:
        if run_id is None:
            return f"{self.session_id}/{job_name}"
        return f"{self.session_id}/{job_name}/{run_id}"

    def mount_pairs(self) -> list[dict[str, str]]:
        return [
            {
                "source": self.input_container,
                "target": self.input_mount_path,
            },
            {
                "source": self.output_container,
                "target": self.output_mount_path,
            },
            {
                "source": self.logs_container,
                "target": self.logs_mount_path,
            },
        ]

    def remote_input_dir(self, run_id: str) -> str:
        generation_name = format_generation_name(
            parse_generation_from_run_id(run_id)
        )
        return f"input/{self.session_id}/{generation_name}/{run_id}"

    def remote_output_dir(self, run_id: str) -> str:
        generation_name = format_generation_name(
            parse_generation_from_run_id(run_id)
        )
        return f"output/{self.session_id}/{generation_name}/{run_id}"
