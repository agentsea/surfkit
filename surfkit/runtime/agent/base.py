from typing import List, TypeVar, Type, Generic, Union, Iterator, Optional
from abc import ABC, abstractmethod

from pydantic import BaseModel
from taskara.models import SolveTaskModel

from ...types import AgentType

R = TypeVar("R", bound="AgentRuntime")
C = TypeVar("C", bound="BaseModel")


class AgentRuntime(Generic[R, C], ABC):

    @classmethod
    def name(cls) -> str:
        return cls.__name__

    @classmethod
    @abstractmethod
    def connect_config_type(cls) -> Type[C]:
        pass

    @classmethod
    @abstractmethod
    def connect(cls, cfg: C) -> R:
        pass

    @abstractmethod
    def run(
        self,
        agent_type: AgentType,
        name: str,
        version: Optional[str] = None,
        env_vars: Optional[dict] = None,
        llm_providers_local: bool = False,
    ) -> None:
        pass

    @abstractmethod
    def solve_task(
        self, agent_name: str, task: SolveTaskModel, follow_logs: bool = False
    ) -> None:
        pass

    @abstractmethod
    def list(self) -> List[str]:
        pass

    @abstractmethod
    def proxy(
        self, name: str, local_port: int, pod_port: int = 8000, background: bool = True
    ) -> None:
        pass

    @abstractmethod
    def delete(self, name: str) -> None:
        pass

    @abstractmethod
    def clean(self) -> None:
        pass

    @abstractmethod
    def logs(self, name: str, follow: bool = False) -> Union[str, Iterator[str]]:
        """
        Fetches the logs from the specified pod.

        Parameters:
            name (str): The name of the pod.

        Returns:
            str: The logs from the pod.
        """
        pass
