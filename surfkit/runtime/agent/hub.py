from typing import List, Type, Union, Iterator, Optional

from pydantic import BaseModel
from taskara import V1Task

from surfkit.types import AgentType
from surfkit.models import V1AgentInstance, V1SolveTask
from .base import AgentRuntime, AgentInstance


class ConnectConfig(BaseModel):
    timeout: Optional[int] = None


class HubAgentRuntime(AgentRuntime):

    @classmethod
    def name(cls) -> str:
        return cls.__name__

    @classmethod
    def connect_config_type(cls) -> Type[ConnectConfig]:
        return ConnectConfig

    @classmethod
    def connect(cls, cfg: ConnectConfig) -> "HubAgentRuntime":
        pass

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

    def solve_task(
        self,
        agent_name: str,
        task: V1SolveTask,
        follow_logs: bool = False,
        attach: bool = False,
    ) -> None:
        pass

    def list(self) -> List[AgentInstance]:
        pass

    def get(self, name: str) -> AgentInstance:
        pass

    def proxy(
        self,
        name: str,
        local_port: Optional[int] = None,
        pod_port: int = 8000,
        background: bool = True,
    ) -> None:
        pass

    def delete(self, name: str) -> None:
        pass

    def clean(self) -> None:
        pass

    def logs(self, name: str, follow: bool = False) -> Union[str, Iterator[str]]:
        """
        Fetches the logs from the specified pod.

        Parameters:
            name (str): The name of the pod.

        Returns:
            str: The logs from the pod.
        """
        pass
