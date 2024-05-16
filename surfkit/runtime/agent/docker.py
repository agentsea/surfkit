import os
import signal
import sys
from typing import Dict, Iterator, List, Optional, Type, Union

import docker
import requests
from agentdesk.util import find_open_port
from docker.errors import NotFound
from mllm import Router
from pydantic import BaseModel

from surfkit.server.models import V1AgentType, V1SolveTask
from surfkit.types import AgentType

from .base import AgentInstance, AgentRuntime


class DockerConnectConfig(BaseModel):
    timeout: Optional[int] = None


class DockerAgentRuntime(AgentRuntime["DockerAgentRuntime", DockerConnectConfig]):

    def __init__(self, cfg: Optional[DockerConnectConfig] = None) -> None:
        if not cfg:
            cfg = DockerConnectConfig()

        self._cfg = cfg
        if cfg.timeout:
            self.client = docker.from_env(timeout=cfg.timeout)
        else:
            self.client = docker.from_env()

    @classmethod
    def name(cls) -> str:
        return "docker"

    @classmethod
    def connect_config_type(cls) -> Type[DockerConnectConfig]:
        return DockerConnectConfig

    def connect_config(self) -> DockerConnectConfig:
        return self._cfg

    @classmethod
    def connect(cls, cfg: DockerConnectConfig) -> "DockerAgentRuntime":
        return cls(cfg)

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
    ) -> AgentInstance:
        labels = {
            "provisioner": "surfkit",
            "agent_type": agent_type.name,
            "agent_name": name,
            "agent_type_model": agent_type.to_v1().model_dump_json(),
        }

        port = find_open_port(9090, 10090)
        if not port:
            raise ValueError("Could not find open port")
        print("running container")

        if not env_vars:
            env_vars = {}
        if llm_providers_local:
            if not agent_type.llm_providers:
                raise ValueError(
                    "no llm providers in agent type, yet llm_providers_local is True"
                )
            found = {}
            for provider_name in agent_type.llm_providers.preference:
                api_key_env = Router.provider_api_keys.get(provider_name)
                if not api_key_env:
                    raise ValueError(f"no api key env for provider {provider_name}")
                key = os.getenv(api_key_env)
                if not key:
                    print("no api key found locally for provider: ", provider_name)
                    continue

                print("api key found locally for provider: ", provider_name)
                found[api_key_env] = key

            if not found:
                raise ValueError(
                    "no api keys found locally for any of the providers in the agent type"
                )
            env_vars.update(found)

        if not agent_type.versions:
            raise ValueError("No versions specified in agent type")

        if not version:
            version = list(agent_type.versions.keys())[0]

        env_vars["SURF_PORT"] = str(port)

        img = agent_type.versions.get(version)
        if not img:
            raise ValueError("img not found")
        container = self.client.containers.run(
            img,
            network_mode="host",
            environment=env_vars,
            detach=True,
            labels=labels,
            name=name,
        )
        if container and type(container) != bytes:
            print(f"ran container '{container.id}'")  # type: ignore

        return AgentInstance(
            name=name,
            type=agent_type,
            runtime=self,
            version=version,
            status="running",
            port=port,
            owner_id=owner_id,
        )

    def solve_task(
        self,
        name: str,
        task: V1SolveTask,
        follow_logs: bool = False,
        attach: bool = False,
        owner_id: Optional[str] = None,
    ) -> None:
        instances = AgentInstance.find(name=name, owner_id=owner_id)
        if not instances:
            raise ValueError(f"No instances found for name '{name}'")
        instance = instances[0]

        print(f"Container '{name}' found.")
        response = requests.post(
            f"http://localhost:{instance.port}/v1/tasks",
            json=task.model_dump(),
        )
        print(f"Task posted with response: {response.status_code}, {response.text}")

        if follow_logs:
            print(f"Following logs for '{name}':")
            self._handle_logs_with_attach(name, attach)

    def _handle_logs_with_attach(self, agent_name: str, attach: bool):
        if attach:
            # Setup the signal handler to catch interrupt signals
            signal.signal(signal.SIGINT, self._signal_handler(agent_name))

        try:
            for line in self.logs(agent_name, follow=True):
                print(line)
        except KeyboardInterrupt:
            # This block will be executed if SIGINT is caught
            print(f"Interrupt received, stopping logs for '{agent_name}'")
            self.delete(agent_name)
        except Exception as e:
            print(f"Error while streaming logs: {e}")

    def _signal_handler(self, agent_name: str):
        def handle_signal(signum, frame):
            print(f"Signal {signum} received, stopping container '{agent_name}'")
            self.delete(agent_name)
            sys.exit(1)

        return handle_signal

    def requires_proxy(self) -> bool:
        """Whether this runtime requires a proxy to be used"""
        return False

    def proxy(
        self,
        name: str,
        local_port: Optional[int] = None,
        agent_port: int = 9090,
        background: bool = True,
        owner_id: Optional[str] = None,
    ) -> Optional[int]:
        print("no proxy needed")
        return

    def list(
        self, owner_id: Optional[str] = None, source: bool = False
    ) -> List[AgentInstance]:

        instances = []
        if source:
            label_filter = {"label": "provisioner=surfkit"}
            containers = self.client.containers.list(filters=label_filter)

            for container in containers:
                agent_type_model = container.labels.get("agent_type_model")
                if not agent_type_model:
                    continue  # Skip containers where the agent type model is not found

                agentv1 = V1AgentType.model_validate_json(agent_type_model)
                agent_type = AgentType.from_v1(agentv1)
                agent_name = container.name

                # Extract the SURF_PORT environment variable
                env_vars = container.attrs.get("Config", {}).get("Env", [])
                port = next(
                    (
                        int(var.split("=")[1])
                        for var in env_vars
                        if var.startswith("SURF_PORT=")
                    ),
                    9090,
                )

                instance = AgentInstance(
                    name=agent_name,
                    type=agent_type,
                    runtime=self,
                    port=port,
                    status="running",
                    owner_id=owner_id,
                )
                instances.append(instance)
        else:
            return AgentInstance.find(owner_id=owner_id, runtime_name=self.name())

        return instances

    def get(
        self, name: str, owner_id: Optional[str] = None, source: bool = False
    ) -> AgentInstance:
        if source:
            try:
                container = self.client.containers.get(name)
                agent_type_model = container.labels.get("agent_type_model")
                if not agent_type_model:
                    raise ValueError("Expected agent type model in labels")

                agentv1 = V1AgentType.model_validate_json(agent_type_model)
                agent_type = AgentType.from_v1(agentv1)

                # Extract the SURF_PORT environment variable
                env_vars = container.attrs.get("Config", {}).get("Env", [])
                port = next(
                    (
                        int(var.split("=")[1])
                        for var in env_vars
                        if var.startswith("SURF_PORT=")
                    ),
                    9090,
                )

                return AgentInstance(
                    name=name,
                    type=agent_type,
                    runtime=self,
                    status="running",
                    port=port,
                    owner_id=owner_id,
                )
            except NotFound:
                raise ValueError(f"Container '{name}' not found")

        else:
            instances = AgentInstance.find(
                name=name, owner_id=owner_id, runtime_name=self.name()
            )
            if not instances:
                raise ValueError(f"Agent instance '{name}' not found")
            return instances[0]

    def delete(self, name: str, owner_id: Optional[str] = None) -> None:
        try:
            # Attempt to get the container by name
            container = self.client.containers.get(name)

            # If found, remove the container
            container.remove(force=True)  # type: ignore
            print(f"Successfully deleted container: {name}")
        except NotFound:
            # Handle the case where the container does not exist
            print(f"Container '{name}' does not exist.")
            raise
        except Exception as e:
            # Handle other potential errors
            print(f"Failed to delete container '{name}': {e}")
            raise

    def clean(self, owner_id: Optional[str] = None) -> None:
        # Define the filter for containers with the specific label
        label_filter = {"label": ["provisioner=surfkit"]}

        # Use the filter to list containers
        containers = self.client.containers.list(filters=label_filter, all=True)

        # Initialize a list to keep track of deleted container names or IDs
        deleted_containers = []

        for container in containers:
            try:
                container_name_or_id = (
                    container.name  # type: ignore
                )  # or container.id for container ID
                container.remove(force=True)  # type: ignore
                print(f"Deleted container: {container_name_or_id}")
                deleted_containers.append(container_name_or_id)
            except Exception as e:
                print(f"Failed to delete container: {e}")

        return None

    def logs(
        self, name: str, follow: bool = False, owner_id: Optional[str] = None
    ) -> Union[str, Iterator[str]]:
        """
        Fetches the logs from the specified container. Can return all logs as a single string,
        or stream the logs as a generator of strings.

        Parameters:
            name (str): The name of the container.
            follow (bool): Whether to continuously follow the logs.

        Returns:
            Union[str, Iterator[str]]: All logs as a single string, or a generator that yields log lines.
        """
        try:
            container = self.client.containers.get(name)
            if follow:
                log_stream = container.logs(stream=True, follow=True)  # type: ignore
                return (line.decode("utf-8").strip() for line in log_stream)
            else:
                return container.logs().decode("utf-8")  # type: ignore
        except NotFound:
            print(f"Container '{name}' does not exist.")
            raise
        except Exception as e:
            print(f"Failed to fetch logs for container '{name}': {e}")
            raise
