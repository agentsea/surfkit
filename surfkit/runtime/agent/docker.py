from typing import List, Optional, Type, Union, Iterator
import os

import docker
from docker.errors import NotFound
from taskara.models import SolveTaskModel
from agentdesk.util import find_open_port
import requests
from pydantic import BaseModel

from .base import AgentRuntime, AgentInstance
from surfkit.types import AgentType
from surfkit.llm import LLMProvider


class ConnectConfig(BaseModel):
    timeout: Optional[int] = None


class DockerAgentRuntime(AgentRuntime):

    def __init__(self, config: ConnectConfig) -> None:
        if config.timeout:
            self.client = docker.from_env(timeout=config.timeout)
        else:
            self.client = docker.from_env()

    @classmethod
    def name(cls) -> str:
        return "docker"

    @classmethod
    def connect_config_type(cls) -> Type[ConnectConfig]:
        return ConnectConfig

    @classmethod
    def connect(cls, cfg: ConnectConfig) -> "DockerAgentRuntime":
        return cls(cfg)

    def run(
        self,
        agent_type: AgentType,
        name: str,
        version: Optional[str] = None,
        env_vars: Optional[dict] = None,
        llm_providers_local: bool = False,
    ) -> AgentInstance:
        labels = {
            "provisioner": "surfkit",
            "agent_type": agent_type.name,
            "agent_name": name,
        }

        port = find_open_port(8001, 9000)
        print("running container")

        if llm_providers_local:
            if not agent_type.llm_providers:
                raise ValueError(
                    "no llm providers in agent type, yet llm_providers_local is True"
                )
            if not env_vars:
                env_vars = {}
            found = {}
            for provider_name in agent_type.llm_providers.preference:
                api_key_env = LLMProvider.provider_api_keys.get(provider_name)
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

        img = agent_type.image
        if version:
            if not agent_type.versions:
                raise ValueError("version supplied but no versions in type")
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

        return AgentInstance(name, agent_type, self)

    def solve_task(
        self, agent_name: str, task: SolveTaskModel, follow_logs: bool = False
    ) -> None:
        try:
            container = self.client.containers.get(agent_name)
            print(f"Container '{agent_name}' found.")
            response = requests.post(
                f"http://localhost:{container.attrs['NetworkSettings']['Ports']['9090/tcp'][0]['HostPort']}/v1/tasks",  # type: ignore
                json=task.model_dump(),
            )
            print(f"Task posted with response: {response.status_code}, {response.text}")
        except NotFound:
            print(f"Container '{agent_name}' does not exist.")
            raise
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            raise

        if follow_logs:
            print(f"Following logs for '{agent_name}':")
            try:
                for line in self.logs(agent_name, follow=True):
                    print(line)
            except Exception as e:
                print(f"Error while streaming logs: {e}")

    def proxy(
        self,
        name: str,
        local_port: Optional[int] = None,
        pod_port: int = 9090,
        background: bool = True,
    ) -> None:
        print("no proxy needed")
        return

    def list(self) -> List[str]:
        label_filter = {"label": ["provisioner=surfkit"]}
        containers = self.client.containers.list(filters=label_filter)

        container_names_or_ids = [container.name for container in containers]  # type: ignore

        return container_names_or_ids

    def delete(self, name: str) -> None:
        try:
            # Attempt to get the container by name
            container = self.client.containers.get(name)

            # If found, remove the container
            container.remove(force=True)  # type: ignore
            print(f"Successfully deleted container: {name}")
        except NotFound:
            # Handle the case where the container does not exist
            print(f"Container '{name}' does not exist.")
        except Exception as e:
            # Handle other potential errors
            print(f"Failed to delete container '{name}': {e}")

    def clean(self) -> None:
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

    def logs(self, name: str, follow: bool = False) -> Union[str, Iterator[str]]:
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
