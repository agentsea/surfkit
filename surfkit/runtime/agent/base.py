from typing import List, TypeVar, Type, Generic, Union, Iterator, Optional
from abc import ABC, abstractmethod

from pydantic import BaseModel
from taskara import V1Task

from surfkit.types import AgentType
from surfkit.models import V1AgentInstance

R = TypeVar("R", bound="AgentRuntime")
C = TypeVar("C", bound="BaseModel")


class AgentInstance:
    """A running agent instance"""

    def __init__(
        self, name: str, type: AgentType, runtime: "AgentRuntime", port: int = 9090
    ) -> None:
        self._runtime = runtime
        self._type = type
        self._name = name
        self._port = port

    @property
    def type(self) -> AgentType:
        return self._type

    @property
    def name(self) -> str:
        return self._name

    @property
    def runtime(self) -> "AgentRuntime":
        return self._runtime

    @property
    def port(self) -> int:
        return self._port

    def proxy(
        self,
        local_port: Optional[int] = None,
        pod_port: int = 9090,
        background: bool = True,
    ) -> None:
        return self._runtime.proxy(self._name, local_port, pod_port, background)

    def solve_task(self, task: V1Task, follow_logs: bool = False) -> None:
        return self._runtime.solve_task(self._name, task, follow_logs)

    def delete(self) -> None:
        return self._runtime.delete(self._name)

    def logs(self, follow: bool = False) -> Union[str, Iterator[str]]:
        """
        Fetches the logs from the specified pod.

        Parameters:
            follow (bool): If True, stream logs until the connection

        Returns:
            str: The logs from the pod.
        """
        return self._runtime.logs(self._name, follow)

    def to_v1(self) -> V1AgentInstance:
        """Convert to V1 API model"""
        return V1AgentInstance(
            name=self._name,
            type=self._type.to_v1(),
            runtime=self._runtime.name(),
            port=self._port,
        )


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
        owner_id: Optional[str] = None,
    ) -> AgentInstance:
        pass

    @abstractmethod
    def solve_task(
        self, agent_name: str, task: V1Task, follow_logs: bool = False
    ) -> None:
        pass

    @abstractmethod
    def list(self) -> List[AgentInstance]:
        pass

    @abstractmethod
    def get(self, name: str) -> AgentInstance:
        pass

    @abstractmethod
    def proxy(
        self,
        name: str,
        local_port: Optional[int] = None,
        pod_port: int = 8000,
        background: bool = True,
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
