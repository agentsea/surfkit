from typing import Optional, List

from pydantic import BaseModel

from .docker import DockerAgentRuntime, ConnectConfig as DockerConnectConfig
from .kube import KubernetesAgentRuntime, ConnectConfig as KubeConnectConfig
from .process import ProcessAgentRuntime, ConnectConfig as ProcessConnectConfig
from .base import AgentRuntime


class AgentRuntimeConfig(BaseModel):
    provider: Optional[str] = None
    docker_config: Optional[DockerConnectConfig] = None
    kube_config: Optional[KubeConnectConfig] = None
    process_config: Optional[ProcessConnectConfig] = None
    preference: List[str] = ["kube", "docker"]


def load_agent_runtime(cfg: AgentRuntimeConfig) -> AgentRuntime:
    for pref in cfg.preference:
        if pref == KubernetesAgentRuntime.name() and cfg.kube_config:
            return KubernetesAgentRuntime.connect(cfg.kube_config)
        elif pref == DockerAgentRuntime.name() and cfg.docker_config:
            return DockerAgentRuntime.connect(cfg.docker_config)
        elif pref == ProcessAgentRuntime.name() and cfg.process_config:
            return ProcessAgentRuntime.connect(cfg.process_config)
    raise ValueError(f"Unknown provider: {cfg.provider}")
