import json
import time
import uuid
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Generic, Iterator, List, Optional, Type, TypeVar, Union

from mllm import Router
from pydantic import BaseModel

from surfkit.db.conn import WithDB
from surfkit.db.models import AgentInstanceRecord
from surfkit.server.models import (
    V1AgentInstance,
    V1AgentType,
    V1RuntimeConnect,
    V1SolveTask,
)
from surfkit.types import AgentType

R = TypeVar("R", bound="AgentRuntime")
C = TypeVar("C", bound="BaseModel")


class AgentStatus(Enum):
    DEFINED = "defined"
    CREATING = "creating"
    RUNNING = "running"
    STOPPED = "stopped"
    DELETING = "deleting"
    DELETED = "deleted"
    ERROR = "error"
    FAILED = "failed"
    RESTARTING = "restarting"


class AgentInstance(WithDB):
    """A running agent instance"""

    def __init__(
        self,
        name: str,
        type: AgentType,
        runtime: "AgentRuntime",
        status: AgentStatus = AgentStatus.DEFINED,
        version: Optional[str] = None,
        port: int = 9090,
        tags: List[str] = [],
        labels: Dict[str, str] = {},
        owner_id: Optional[str] = None,
        icon: Optional[str] = None,
    ) -> None:
        self._id = str(uuid.uuid4())
        self._runtime = runtime
        self._type = type
        self._name = name
        self._version = version
        self._port = port
        self._tags = tags
        self._labels = labels
        self._status = status
        self._owner_id = owner_id
        self._icon = icon
        self._created = time.time()
        self._updated = time.time()

        self.save()

    @property
    def id(self) -> str:
        return self._id

    @property
    def status(self) -> AgentStatus:
        return self._status

    @status.setter
    def status(self, value: AgentStatus) -> None:
        self._status = value
        self.save()

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
    def version(self) -> Optional[str]:
        return self._version

    @property
    def port(self) -> int:
        return self._port  # type: ignore

    @property
    def tags(self) -> List[str]:
        return self._tags

    @property
    def labels(self) -> Dict[str, str]:
        return self._labels

    @property
    def owner_id(self) -> Optional[str]:
        return self._owner_id  # type: ignore

    @property
    def created(self) -> float:
        return self._created  # type: ignore

    @property
    def updated(self) -> float:
        return self._updated  # type: ignore

    @property
    def icon(self) -> Optional[str]:
        return self._icon

    def proxy(
        self,
        local_port: Optional[int] = None,
        background: bool = True,
    ) -> Optional[int]:
        """Proxy the agent instance locally.

        Args:
            local_port (Optional[int], optional): Local port to use. Defaults to None.
            background (bool, optional): Whether to run in the background. Defaults to True.

        Returns:
            Optional[int]: Optional pid of the proxy
        """
        return self._runtime.proxy(self._name, local_port, self.port, background)

    def solve_task(self, task: V1SolveTask, follow_logs: bool = False) -> None:
        """Solve a task with the agent instance.

        Args:
            task (V1SolveTask): The task
            follow_logs (bool, optional): Whether to follow the logs. Defaults to False.
        """
        return self._runtime.solve_task(self._name, task, follow_logs)

    @classmethod
    def active_runtimes(cls) -> List["AgentRuntime"]:
        """Find all unique runtime objects currently used by agent instances.

        Returns:
            List[AgentRuntime]: A list of unique runtime objects.
        """
        unique_runtimes = set()
        runtime_objects = []

        for db in cls.get_db():
            records = (
                db.query(
                    AgentInstanceRecord.runtime_name, AgentInstanceRecord.runtime_config
                )
                .distinct()
                .all()
            )
            unique_runtimes.update(
                (record.runtime_name, record.runtime_config) for record in records
            )

        # Import the function to convert runtime names to runtime objects
        from surfkit.runtime.agent.load import runtime_from_name

        for runtime_name, runtime_config in unique_runtimes:
            try:
                runtype = runtime_from_name(runtime_name)
                runcfg = runtype.connect_config_type().model_validate_json(
                    str(runtime_config)
                )
                runtime = runtype.connect(runcfg)
                runtime_objects.append(runtime)
            except ValueError:
                continue

        return runtime_objects

    def delete(self, force: bool = False) -> None:
        """Deletes the agent instance from the runtime and the database.

        Args:
            force (bool, optional): Whether to force delete. Defaults to False.
        """
        # First, delete the agent instance from the runtime.
        try:
            self._runtime.delete(self._name)
        except Exception as e:
            if not force:
                raise e

        # After the runtime deletion, proceed to delete the record from the database.
        for db in self.get_db():
            record = db.query(AgentInstanceRecord).filter_by(id=self._id).one()
            db.delete(record)
            db.commit()

    def logs(self, follow: bool = False) -> Union[str, Iterator[str]]:
        """
        Fetches the logs from the specified pod.

        Parameters:
            follow (bool): If True, stream logs until the connection

        Returns:
            str: The logs from the pod.
        """
        return self._runtime.logs(self._name, follow)

    def save(self) -> None:
        """Save the agent instance"""
        for db in self.get_db():
            record = self.to_record()
            db.merge(record)
            db.commit()

    @classmethod
    def find(cls, **kwargs) -> List["AgentInstance"]:
        """Find agent instances by name and type.

        Returns:
            List[AgentInstance]: A list of agent instances.
        """
        for db in cls.get_db():
            records = (
                db.query(AgentInstanceRecord)
                .filter_by(**kwargs)
                .order_by(AgentInstanceRecord.created.desc())
                .all()
            )
            return [cls.from_record(record) for record in records]
        raise ValueError("No session")

    def to_v1(self) -> V1AgentInstance:
        """Convert to V1 API model"""
        return V1AgentInstance(
            name=self._name,
            type=self._type.to_v1(),
            runtime=V1RuntimeConnect(
                name=self._runtime.name(), connect_config=self.runtime.connect_config()
            ),
            version=self._version,
            port=self._port,  # type: ignore
            tags=self._tags,
            labels=self._labels,
            status=self._status.value,
            owner_id=self._owner_id,  # type: ignore
            icon=self._icon,
            created=self._created,  # type: ignore
            updated=self._updated,  # type: ignore
        )

    def to_record(self) -> AgentInstanceRecord:
        """Convert to DB model"""
        runtime_cfg = self._runtime.connect_config().model_dump_json()

        return AgentInstanceRecord(
            id=self._id,
            name=self._name,
            type=self._type.to_v1().model_dump_json(),
            runtime_name=self._runtime.name(),
            runtime_config=runtime_cfg,
            version=self._version,
            port=self._port,
            tags=json.dumps(self.tags),
            labels=json.dumps(self.labels),
            status=self._status.value,
            owner_id=self._owner_id,
            icon=self._icon,
            created=self._created,
            updated=self._updated,
        )

    @classmethod
    def from_record(cls, record: AgentInstanceRecord) -> "AgentInstance":
        from surfkit.runtime.agent.load import runtime_from_name

        runtype = runtime_from_name(str(record.runtime_name))
        runcfg = runtype.connect_config_type().model_validate_json(
            str(record.runtime_config)
        )
        runtime = runtype.connect(runcfg)

        obj = cls.__new__(cls)
        obj._id = str(record.id)
        obj._name = str(record.name)
        obj._type = AgentType.from_v1(V1AgentType.model_validate_json(str(record.type)))
        obj._runtime = runtime
        obj._version = str(record.version)
        obj._status = AgentStatus(record.status)
        obj._port = record.port
        obj._tags = json.loads(str(record.tags))
        obj._labels = json.loads(str(record.labels))
        obj._owner_id = record.owner_id
        obj._icon = record.icon
        obj._created = record.created
        obj._updated = record.updated

        return obj

    def refresh(self) -> None:
        """
        Refresh the agent instance's fields to reflect the current state in the database.
        """
        for db in self.get_db():
            record = db.query(AgentInstanceRecord).filter_by(id=self._id).one_or_none()
            if record:
                self._name = str(record.name)
                self._type = AgentType.from_v1(
                    V1AgentType.model_validate_json(str(record.type))
                )
                self._version = str(record.version)
                self._port = record.port
                self._tags = json.loads(str(record.tags))
                self._labels = json.loads(str(record.labels))
                self._status = AgentStatus(record.status)
                self._owner_id = record.owner_id
                self._created = record.created
                self._updated = record.updated
            else:
                raise ValueError(
                    f"AgentInstance with id {self._id} not found in the database."
                )


class AgentRuntime(Generic[R, C], ABC):

    @classmethod
    def name(cls) -> str:
        return cls.__name__

    @classmethod
    @abstractmethod
    def connect_config_type(cls) -> Type[C]:
        """The pydantic model which defines the schema for connecting to this runtime

        Returns:
            Type[C]: The type
        """
        pass

    @abstractmethod
    def connect_config(cls) -> C:
        """The connect config for this runtime instance

        Returns:
            C: Connect config
        """
        pass

    @classmethod
    @abstractmethod
    def connect(cls, cfg: C) -> R:
        """Connect to the runtime using this configuration

        Args:
            cfg (C): Connect config

        Returns:
            R: A runtime
        """
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
        tags: Optional[List[str]] = None,
        labels: Optional[Dict[str, str]] = None,
        auth_enabled: bool = True,
        debug: bool = False,
    ) -> AgentInstance:
        """Run the agent

        Args:
            agent_type (AgentType): The type of agent
            name (str): Name of the agent
            version (Optional[str], optional): Version to use. Defaults to None.
            env_vars (Optional[dict], optional): Env vars to supply. Defaults to None.
            llm_providers_local (bool, optional): Whether to copy local llm providers. Defaults to False.
            owner_id (Optional[str], optional): Owner ID. Defaults to None.
            tags (Optional[List[str]], optional): Tags for the agent. Defaults to None.
            labels (Optional[Dict[str, str]], optional): Labels for the agent. Defaults to None.
            auth_enabled (bool, optional): Whether to enable auth. Defaults to True.
            debug (bool, optional): Whether to run in debug mode. Defaults to False.

        Returns:
            AgentInstance: An agent instance
        """
        pass

    @abstractmethod
    def solve_task(
        self,
        name: str,
        task: V1SolveTask,
        follow_logs: bool = False,
        attach: bool = False,
        owner_id: Optional[str] = None,
    ) -> None:
        """Solve a task with an agent

        Args:
            name (str): Name of the agent
            task (V1SolveTask): The task
            follow_logs (bool, optional): Whether to follow the logs. Defaults to False.
            attach (bool, optional): Whether to attache the current process to the agent
                If this process dies the agent will also die. Defaults to False.
            owner_id (Optional[str], optional): Optional owner ID. Defaults to None.
        """
        pass

    @abstractmethod
    def list(
        self, owner_id: Optional[str] = None, source: bool = False
    ) -> List[AgentInstance]:
        """List agent instances

        Args:
            owner_id (Optional[str], optional): An optional owner id. Defaults to None.
            source (bool, optional): Whether to list directly from the source. Defaults to False.

        Returns:
            List[AgentInstance]: A list of agent instances
        """
        pass

    @abstractmethod
    def get(
        self, name: str, owner_id: Optional[str] = None, source: bool = False
    ) -> AgentInstance:
        """Get an agent instance

        Args:
            name (str): Name of the agent
            owner_id (Optional[str], optional): Optional owner ID. Defaults to None.
            source (bool, optional): Whether to fetch directly from the source. Defaults to False.

        Returns:
            AgentInstance: An agent instance
        """
        pass

    @abstractmethod
    def requires_proxy(self) -> bool:
        """Whether this runtime requires a proxy to be used"""
        pass

    @abstractmethod
    def proxy(
        self,
        name: str,
        local_port: Optional[int] = None,
        agent_port: int = 9090,
        background: bool = True,
        owner_id: Optional[str] = None,
    ) -> Optional[int]:
        """Proxy a port to the agent

        Args:
            name (str): Name of the agent
            local_port (Optional[int], optional): Local port to proxy to. Defaults to None.
            agent_port (int, optional): The agents port. Defaults to 9090.
            background (bool, optional): Whether to run the proxy in the background. Defaults to True.
            owner_id (Optional[str], optional): An optional owner ID. Defaults to None.

        Returns:
                Optional[int]: The pid of the proxy
        """
        pass

    @abstractmethod
    def delete(self, name: str, owner_id: Optional[str] = None) -> None:
        """Delete an agent instance

        Args:
            name (str): Name of the agent
            owner_id (Optional[str], optional): An optional owner id. Defaults to None.
        """
        pass

    @abstractmethod
    def clean(self, owner_id: Optional[str] = None) -> None:
        """Delete all agent instances

        Args:
            owner_id (Optional[str], optional): An optional owner ID to scope it to. Defaults to None.
        """
        pass

    @abstractmethod
    def logs(
        self, name: str, follow: bool = False, owner_id: Optional[str] = None
    ) -> Union[str, Iterator[str]]:
        """
        Fetches the logs from the specified pod.

        Parameters:
            name (str): The name of the pod.

        Returns:
            str: The logs from the pod.
        """
        pass

    @abstractmethod
    def refresh(self, owner_id: Optional[str] = None) -> None:
        """
        Synchronizes the state between running processes and the database.
        Ensures that the processes and the database reflect the same set of running agent instances.

        Parameters:
            owner_id (Optional[str]): The ID of the owner to filter instances.
        """
        pass

    @abstractmethod
    def runtime_local_addr(self, name: str, owner_id: Optional[str] = None) -> str:
        """
        Returns the local address of the agent with respect to the runtime
        """
        pass

    def check_llm_providers(self, agent_type: AgentType, environment: dict) -> None:
        if agent_type.llm_providers:
            found = 0
            num_opts = len(agent_type.llm_providers.preference)
            key_opts = []
            if not num_opts:
                return
            for provider in agent_type.llm_providers.preference:
                key = Router.provider_api_keys.get(provider)
                if not key:
                    continue
                key_opts.append(key)
                if environment.get(key):
                    found += 1
            if not found:
                raise ValueError(
                    (
                        "No API keys found for any of the LLM providers required for this agent type."
                        f" Options are {', '.join(set(key_opts))}"
                    )
                )
