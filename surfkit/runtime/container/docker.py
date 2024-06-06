from typing import List, Optional, Tuple, Type, TypeVar

import docker
import requests
from agentdesk.util import find_open_port
from docker.errors import NotFound
from namesgenerator import get_random_name
from pydantic import BaseModel

from .base import ContainerRuntime


class ConnectConfig(BaseModel):
    pass


class DockerRuntime(ContainerRuntime):
    """A container runtime that uses docker"""

    def __init__(self) -> None:
        self.client = docker.from_env()

    @classmethod
    def name(cls) -> str:
        return "docker"

    @classmethod
    def connect_config_type(cls) -> Type[ConnectConfig]:
        return ConnectConfig

    @classmethod
    def connect(cls, cfg: ConnectConfig) -> "DockerRuntime":
        return cls()

    def create(
        self,
        image: str,
        name: Optional[str] = None,
        env_vars: Optional[dict] = None,
        mem_request: Optional[str] = "500m",
        mem_limit: Optional[str] = "2Gi",
        cpu_request: Optional[str] = "1",
        cpu_limit: Optional[str] = "4",
        gpu_mem: Optional[str] = None,
    ) -> None:
        if not name:
            name = get_random_name("-")

        labels = {
            "provisioner": "surfkit",
        }

        port = find_open_port(9090, 10090)
        print("running container")
        container = self.client.containers.run(
            image,
            network_mode="host",
            environment=env_vars,
            detach=True,
            labels=labels,
            name=name,
            mem_limit=mem_limit,
            mem_reservation=mem_request,
            nano_cpus=int(float(cpu_limit) * 1e9),  # type: ignore
        )
        if container and type(container) != bytes:
            print(f"ran container '{container.id}'")  # type: ignore

    def call(
        self,
        name: str,
        path: str,
        method: str,
        port: int = 8080,
        data: Optional[dict] = None,
        headers: Optional[dict] = None,
    ) -> Tuple[int, str]:
        """
        Makes an HTTP request to the specified container.

        Parameters:
            name (str): The name of the container.
            route (str): The endpoint route (e.g., 'api/data').
            method (str): HTTP method (e.g., 'GET', 'POST').
            port (int): The port on which the container's server is listening.
            params (dict, optional): The URL parameters for GET or DELETE requests.
            body (dict, optional): The JSON body for POST, PUT requests.
            headers (dict, optional): HTTP headers.

        Returns:
            requests.Response: The HTTP response.
        """
        try:
            container = self.client.containers.get(name)
            print(f"Container '{name}' found.")
        except NotFound:
            print(f"Container '{name}' does not exist.")
            raise
        except Exception as e:
            print(f"An unexpected error occurred calling docker container: {e}")
            raise

        url = f"http://localhost:{port}{path}"

        # Dynamically calling the method based on 'method' parameter
        http_request = getattr(requests, method.lower(), requests.get)

        if not callable(http_request):
            raise ValueError(f"Unsupported HTTP method: {method}")

        if method.upper() in ["GET", "DELETE"]:  # These methods should use params
            response = http_request(url, params=data, headers=headers)
            return response.status_code, response.text
        else:
            response = http_request(url, json=data, headers=headers)
            return response.status_code, response.text

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
            raise
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

        return

    def logs(self, name: str) -> str:
        """
        Fetches the logs from the specified container.

        Parameters:
            name (str): The name of the container.

        Returns:
            str: The logs from the container.
        """
        try:
            container = self.client.containers.get(name)
            return container.logs().decode("utf-8")  # type: ignore
        except NotFound:
            print(f"Container '{name}' does not exist.")
            raise
        except Exception as e:
            print(f"An unexpected error occurred retrieving logs: {e}")
            raise
