from typing import List, Optional, Type

from pydantic import BaseModel

from surfkit.server.models import V1RuntimeConnect

from .base import AgentRuntime
from .docker import DockerAgentRuntime, DockerConnectConfig
from .kube import KubeAgentRuntime, KubeConnectConfig
from .process import ProcessAgentRuntime, ProcessConnectConfig


class AgentRuntimeConfig(BaseModel):
    provider: Optional[str] = None
    docker_config: Optional[DockerConnectConfig] = None
    kube_config: Optional[KubeConnectConfig] = None
    process_config: Optional[ProcessConnectConfig] = None
    preference: List[str] = ["kube", "docker", "process"]


def runtime_from_name(name: str) -> Type[AgentRuntime]:
    for runt in RUNTIMES:
        if runt.name() == name:
            return runt
    raise ValueError(f"Unknown runtime '{name}'")


def load_agent_runtime(cfg: AgentRuntimeConfig) -> AgentRuntime:
    for pref in cfg.preference:
        if pref == KubeAgentRuntime.name() and cfg.kube_config:
            return KubeAgentRuntime.connect(cfg.kube_config)
        elif pref == DockerAgentRuntime.name() and cfg.docker_config:
            return DockerAgentRuntime.connect(cfg.docker_config)
        elif pref == ProcessAgentRuntime.name() and cfg.process_config:
            return ProcessAgentRuntime.connect(cfg.process_config)
    raise ValueError(f"Unknown provider: {cfg.provider}")


RUNTIMES: List[Type[AgentRuntime]] = [DockerAgentRuntime, KubeAgentRuntime, ProcessAgentRuntime]  # type: ignore


def load_from_connect(connect: V1RuntimeConnect) -> AgentRuntime:
    for runt in RUNTIMES:
        if connect.name == runt.name():
            print("connect config: ", connect.connect_config)
            print("type: ", type(connect.connect_config))
            cfg = runt.connect_config_type().model_validate(connect.connect_config)
            return runt.connect(cfg)

    raise ValueError(f"Unknown runtime: {connect.name}")
