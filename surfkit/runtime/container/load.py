from typing import List, Optional

from pydantic import BaseModel

from .base import ContainerRuntime
from .docker import ConnectConfig as DockerConnectConfig
from .docker import DockerRuntime
from .kube import ConnectConfig as KubeConnectConfig
from .kube import KubernetesRuntime


class ContainerRuntimeConfig(BaseModel):
    provider: Optional[str] = None
    docker_config: Optional[DockerConnectConfig] = None
    kube_config: Optional[KubeConnectConfig] = None
    preference: List[str] = ["kube", "docker"]


def load_container_runtime(cfg: ContainerRuntimeConfig) -> ContainerRuntime:
    if cfg.provider == KubernetesRuntime.name():
        if not cfg.kube_config:
            raise ValueError("Kubernetes config is required")
        return KubernetesRuntime.connect(cfg.kube_config)

    elif cfg.provider == DockerRuntime.name():
        if not cfg.docker_config:
            raise ValueError("Docker config is required")
        return DockerRuntime.connect(cfg.docker_config)

    else:
        raise ValueError(f"Unknown provider: {cfg.provider}")
