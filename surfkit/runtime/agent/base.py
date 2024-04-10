from typing import List, TypeVar
from abc import ABC, abstractmethod

import docker
from taskara.models import SolveTaskModel

from ...types import AgentType


class AgentRuntime(ABC):

    def __init__(self) -> None:
        self.client = docker.from_env()

    @abstractmethod
    def run(self, agent_type: AgentType, name: str) -> None:
        pass

    @abstractmethod
    def solve_task(self, agent_name: str, task: SolveTaskModel) -> None:
        pass

    @abstractmethod
    def list(self) -> List[str]:
        pass

    @abstractmethod
    def delete(self, name: str) -> bool:
        pass

    @abstractmethod
    def clean(self) -> List[str]:
        pass
