import logging
import os
import platform
import socket
import time
from typing import Dict, Iterator, List, Optional, Type, Union

import docker
import requests
from agentdesk.util import find_open_port
from docker.api.client import APIClient
from docker.errors import NotFound
from mllm import Router
from pydantic import BaseModel

from surfkit.server.models import V1AgentType, V1SolveTask
from surfkit.types import AgentType

from .base import AgentInstance, AgentRuntime, AgentStatus
from .util import pull_image

logger = logging.getLogger(__name__)


class DockerConnectConfig(BaseModel):
    timeout: Optional[int] = None


class DockerAgentRuntime(AgentRuntime["DockerAgentRuntime", DockerConnectConfig]):

    def __init__(self, cfg: Optional[DockerConnectConfig] = None) -> None:
        self._configure_docker_socket()
        if not cfg:
            cfg = DockerConnectConfig()

        self._cfg = cfg
        if cfg.timeout:
            self.client = docker.from_env(timeout=cfg.timeout)
        else:
            self.client = docker.from_env()

    def _configure_docker_socket(self):
        if os.path.exists("/var/run/docker.sock"):
            docker_socket = "unix:///var/run/docker.sock"
        else:
            user = os.environ.get("USER")
            if os.path.exists(f"/Users/{user}/.docker/run/docker.sock"):
                docker_socket = f"unix:///Users/{user}/.docker/run/docker.sock"
            else:
                raise FileNotFoundError(
                    (
                        "Neither '/var/run/docker.sock' nor '/Users/<USER>/.docker/run/docker.sock' are available."
                        "Please make sure you have Docker installed and running."
                    )
                )
        os.environ["DOCKER_HOST"] = docker_socket

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

    def ensure_network(self, network_name: str) -> None:
        """Ensure that the specified Docker network exists, creating it if necessary."""
        try:
            self.client.networks.get(network_name)
            print(f"Network '{network_name}' already exists.")
        except NotFound:
            self.client.networks.create(network_name)
            print(f"Network '{network_name}' created.")

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
        """
        Run a Docker container for the specified agent type.

        Args:
            agent_type (AgentType): The type of agent to run.
            name (str): The name of the agent.
            version (Optional[str], optional): The version of the agent. Defaults to None.
            env_vars (Optional[dict], optional): Environment variables for the container. Defaults to None.
            llm_providers_local (bool, optional): Whether to use local LLM providers. Defaults to False.
            owner_id (Optional[str], optional): The ID of the owner. Defaults to None.
            tags (Optional[List[str]], optional): Tags for the container. Defaults to None.
            labels (Optional[Dict[str, str]], optional): Labels for the container. Defaults to None.
            auth_enabled (bool, optional): Whether authentication is enabled. Defaults to True.
            debug (bool, optional): Whether to run in debug mode. Defaults to False.

        Returns:
            AgentInstance: An instance of the running agent.
        """
        labels = {
            "provisioner": "surfkit",
            "agent_type": agent_type.name,
            "agent_name": name,
            "agent_type_model": agent_type.to_v1().model_dump_json(),
        }

        port = find_open_port(9090, 10090)
        if not port:
            raise ValueError("Could not find open port")

        if not env_vars:
            env_vars = {}

        if debug:
            env_vars["DEBUG"] = "true"
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

        self.check_llm_providers(agent_type, env_vars)

        if not agent_type.versions:
            raise ValueError("No versions specified in agent type")

        if not version:
            version = list(agent_type.versions.keys())[0]

        env_vars["SERVER_PORT"] = str(port)
        if not auth_enabled:
            env_vars["AGENT_NO_AUTH"] = "true"

        if agent_type.llm_providers:
            env_vars["MODEL_PREFERENCE"] = ",".join(agent_type.llm_providers.preference)

        img = agent_type.versions.get(version)
        if not img:
            raise ValueError("img not found")

        # Initialize tqdm progress bar
        api_client = APIClient()

        # Pull the image with progress tracking
        pull_image(img, api_client)

        print(f"running image {img}")
        self.ensure_network("agentsea")

        container_params = {
            "image": img,
            "network": "agentsea",
            "ports": {port: port},
            "environment": env_vars,
            "detach": True,
            "labels": labels,
            "name": name,
        }

        # Add extra_hosts only for Linux
        if platform.system() == "Linux":
            container_params["extra_hosts"] = {"host.docker.internal": "host-gateway"}

        try:
            container = self.client.containers.run(**container_params)
        except Exception as e:
            raise RuntimeError(
                f"Could not run container '{name}' for agent type '{agent_type.name}' with version '{version}': {e}"
            )
        if container and type(container) != bytes:
            print(f"container id '{container.id}'")  # type: ignore

        # Wait for the container to be in the "running" state
        for _ in range(10):
            container.reload()  # type: ignore
            if container.status == "running":  # type: ignore
                break
            time.sleep(1)
        else:
            raise RuntimeError(f"Container '{name}' did not start in time")

        # Check /health endpoint
        health_url = f"http://localhost:{port}/health"
        for _ in range(60):
            try:
                print("waiting for agent to be ready...")
                response = requests.get(health_url)
                if response.status_code == 200:
                    print(f"Health check passed for '{name}'")
                    break
            except requests.RequestException as e:
                logger.debug(f"Health check failed: {e}")
            time.sleep(1)
        else:
            container.remove(force=True)  # type: ignore
            raise RuntimeError(f"Container '{name}' did not pass health check")

        return AgentInstance(
            name=name,
            type=agent_type,
            runtime=self,
            version=version,
            status=AgentStatus.RUNNING,
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
        """
        Solve a task by posting it to the specified container instance.

        Args:
            name (str): The name of the container instance.
            task (V1SolveTask): The task to be solved.
            follow_logs (bool, optional): Whether to follow the logs of the container instance. Defaults to False.
            attach (bool, optional): Whether to attach to the container instance. Defaults to False.
            owner_id (str, optional): The ID of the owner. Defaults to None.

        Raises:
            ValueError: If no instances are found for the specified name.

        Returns:
            None
        """
        instances = AgentInstance.find(name=name, owner_id=owner_id)
        if not instances:
            raise ValueError(f"No instances found for name '{name}'")
        instance = instances[0]

        # Determine the appropriate host address based on the platform
        host_address = "host.docker.internal"

        # TODO: This is a hack to make the qemu desktops work with docker agents, should likely be reworked
        if task.task.device and task.task.device.type.lower() == "desktop":
            cfg = task.task.device.config
            if hasattr(cfg, "agentd_url"):
                agentd_url: str = cfg.agentd_url  # type: ignore
                if agentd_url.startswith("http://localhost"):
                    agentd_url = agentd_url.replace(
                        "http://localhost", f"http://{host_address}"
                    )
                    task.task.device.config.agentd_url = agentd_url  # type: ignore
                    logging.debug(f"replaced agentd url: {task.task.device.config}")

                elif agentd_url.startswith("localhost"):
                    agentd_url = agentd_url.replace("localhost", host_address)
                    task.task.device.config.agentd_url = agentd_url  # type: ignore
                    logging.debug(f"replaced agentd url: {task.task.device.config}")

        print(f"Container '{name}' found.")
        response = requests.post(
            f"http://localhost:{instance.port}/v1/tasks",
            json=task.model_dump(),
        )
        print(f"Task posted with response: {response.status_code}, {response.text}")

        if follow_logs:
            print(f"Following logs for '{name}':")
            self._handle_logs_with_attach(name, attach)

    def _get_host_ip(self) -> str:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            # Doesn't even have to be reachable
            s.connect(("10.254.254.254", 1))
            ip = s.getsockname()[0]
        except Exception:
            ip = "127.0.0.1"
        finally:
            s.close()
        return ip

    def runtime_local_addr(self, name: str, owner_id: Optional[str] = None) -> str:
        """
        Returns the local address of the agent with respect to the runtime
        """
        instances = AgentInstance.find(name=name, owner_id=owner_id)
        if not instances:
            raise ValueError(f"No instances found for name '{name}'")
        instance = instances[0]

        return f"http://{name}:{instance.port}"

    def _handle_logs_with_attach(self, agent_name: str, attach: bool):
        import typer

        try:
            for line in self.logs(agent_name, follow=True):
                print(line)
                if line.startswith("► task run ended"):
                    if not attach:
                        print("")
                        stop = typer.confirm(
                            "Task is finished, do you want to stop the agent?"
                        )
                    else:
                        stop = attach

                    if stop:
                        try:
                            instances = AgentInstance.find(name=agent_name)
                            if instances:
                                instances[0].delete(force=True)
                            else:
                                print(f"No instances found for name '{agent_name}'")
                        except:
                            pass
                    return
        except KeyboardInterrupt:
            # This block will be executed if SIGINT is caught
            print(f"Interrupt received, stopping logs for '{agent_name}'")

            if not attach:
                print("")
                stop = typer.confirm("Do you want to stop the agent?")
            else:
                stop = attach
            try:
                if stop:
                    instances = AgentInstance.find(name=agent_name)
                    if instances:
                        instances[0].delete(force=True)
                    else:
                        print(f"No instances found for name '{agent_name}'")
            except:
                pass
        except Exception as e:
            print(f"Error while streaming logs: {e}")

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
        """
        Retrieve a list of AgentInstances.

        Args:
            owner_id (Optional[str]): The owner ID to filter the instances by. Defaults to None.
            source (bool): Flag indicating whether to retrieve instances from the source directly or use the find method.
                Defaults to False.

        Returns:
            List[AgentInstance]: A list of AgentInstance objects.

        """
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

                # Extract the SERVER_PORT environment variable
                env_vars = container.attrs.get("Config", {}).get("Env", [])
                port = next(
                    (
                        int(var.split("=")[1])
                        for var in env_vars
                        if var.startswith("SERVER_PORT=")
                    ),
                    9090,
                )

                instance = AgentInstance(
                    name=agent_name,
                    type=agent_type,
                    runtime=self,
                    port=port,
                    status=AgentStatus.RUNNING,
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

                # Extract the SERVER_PORT environment variable
                env_vars = container.attrs.get("Config", {}).get("Env", [])
                port = next(
                    (
                        int(var.split("=")[1])
                        for var in env_vars
                        if var.startswith("SERVER_PORT=")
                    ),
                    9090,
                )

                return AgentInstance(
                    name=name,
                    type=agent_type,
                    runtime=self,
                    status=AgentStatus.RUNNING,
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
        """
        Deletes a Docker container by name.

        Args:
            name (str): The name of the container to delete.
            owner_id (Optional[str]): The ID of the container's owner (default: None).

        Raises:
            NotFound: If the container does not exist.
            Exception: If an error occurs while deleting the container.

        Returns:
            None
        """
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
                return (line.decode("utf-8").strip() for line in log_stream)  # type: ignore
            else:
                return container.logs().decode("utf-8")  # type: ignore
        except NotFound:
            print(f"Container '{name}' does not exist.")
            raise
        except Exception as e:
            print(f"Failed to fetch logs for container '{name}': {e}")
            raise

    def refresh(self, owner_id: Optional[str] = None) -> None:
        """
        Synchronizes the state between running Docker containers and the database.
        Ensures that the containers and the database reflect the same set of running agent instances.

        Parameters:
            owner_id (Optional[str]): The ID of the owner to filter instances.
        """
        # Fetch the running containers from Docker
        label_filter = {"label": "provisioner=surfkit"}
        running_containers = self.client.containers.list(filters=label_filter)

        # Fetch the agent instances from the database
        db_instances = AgentInstance.find(owner_id=owner_id, runtime_name=self.name())

        # Create a mapping of container names to containers
        running_containers_map = {container.name: container for container in running_containers}  # type: ignore

        # Create a mapping of instance names to instances
        db_instances_map = {instance.name: instance for instance in db_instances}

        # Check for containers that are running but not in the database
        for container_name, container in running_containers_map.items():
            if container_name not in db_instances_map:
                print(
                    f"Container '{container_name}' is running but not in the database. Creating new instance."
                )
                agent_type_model = container.labels.get("agent_type_model")
                if not agent_type_model:
                    print(
                        f"Skipping container '{container_name}' as it lacks 'agent_type_model' label."
                    )
                    continue

                agentv1 = V1AgentType.model_validate_json(agent_type_model)
                agent_type = AgentType.from_v1(agentv1)
                env_vars = container.attrs.get("Config", {}).get("Env", [])
                port = next(
                    (
                        int(var.split("=")[1])
                        for var in env_vars
                        if var.startswith("SERVER_PORT=")
                    ),
                    9090,
                )
                new_instance = AgentInstance(
                    name=container_name,
                    type=agent_type,
                    runtime=self,
                    status=AgentStatus.RUNNING,
                    port=port,
                    owner_id=owner_id,
                )
                new_instance.save()

        # Check for instances in the database that are not running as containers
        for instance_name, instance in db_instances_map.items():
            if instance_name not in running_containers_map:
                print(
                    f"Instance '{instance_name}' is in the database but not running. Removing from database."
                )
                instance.delete(force=True)

        logger.debug(
            "Refresh complete. State synchronized between Docker and the database."
        )
