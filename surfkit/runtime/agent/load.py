from typing import Optional, List

from pydantic import BaseModel

from .docker import DockerAgentRuntime, ConnectConfig as DockerConnectConfig
from .kube import KubernetesAgentRuntime, ConnectConfig as KubeConnectConfig
from .base import AgentRuntime


class AgentRuntimeConfig(BaseModel):
    provider: Optional[str] = None
    docker_config: Optional[DockerConnectConfig] = None
    kube_config: Optional[KubeConnectConfig] = None
    preference: List[str] = ["kube", "docker"]


def load_container_runtime(cfg: AgentRuntimeConfig) -> AgentRuntime:
    if cfg.provider == KubernetesAgentRuntime.name():
        if not cfg.kube_config:
            raise ValueError("Kubernetes config is required")
        return KubernetesAgentRuntime.connect(cfg.kube_config)

    elif cfg.provider == DockerAgentRuntime.name():
        if not cfg.docker_config:
            raise ValueError("Docker config is required")
        return DockerAgentRuntime.connect(cfg.docker_config)

    else:
        raise ValueError(f"Unknown provider: {cfg.provider}")
