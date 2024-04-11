from .docker import DockerRuntime
from .kube import KubernetesRuntime
from .base import ContainerRuntime


def load_runtime(cfg):
    pass
