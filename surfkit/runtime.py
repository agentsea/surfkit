from typing import List

import docker
from docker.errors import NotFound
from taskara.models import SolveTaskModel
from taskara import Task
from agentdesk.util import find_open_port
import requests
from namesgenerator import get_random_name

from .types import AgentType


class DockerAgentRuntime:

    def __init__(self) -> None:
        self.client = docker.from_env()

    def run(self, agent_type: AgentType, name: str) -> None:
        env_vars = {}
        labels = {
            "provisioner": "surfkit",
            "agent_type": agent_type.name,
            "agent_name": name,
        }

        port = find_open_port(8000, 9000)
        print("running container")
        container = self.client.containers.run(
            agent_type.image,
            network_mode="host",
            environment=env_vars,
            detach=True,
            labels=labels,
            name=name,
        )
        print(f"ran container '{container.id}'")

    def solve_task(self, agent_name: str, task: SolveTaskModel) -> Task:
        try:
            container = self.client.containers.get(agent_name)
            print(f"Container '{agent_name}' found.")
        except NotFound:
            print(f"Container '{agent_name}' does not exist.")
            raise
        except Exception as e:
            print(f"An unexpected error occurred connecting to agent container: {e}")
            raise

        requests.post(
            f"http://localhost:8000/v1/tasks",
            json=task.model_dump(),
        )

        for line in container.logs(stream=True, follow=True):
            print(line.decode().strip())

    def list(self) -> List[str]:
        label_filter = {"label": ["provisioner=surfkit"]}
        containers = self.client.containers.list(filters=label_filter)

        container_names_or_ids = [container.name for container in containers]

        return container_names_or_ids

    def delete_by_name(self, name: str) -> bool:
        try:
            # Attempt to get the container by name
            container = self.client.containers.get(name)

            # If found, remove the container
            container.remove(force=True)
            print(f"Successfully deleted container: {name}")
            return True
        except NotFound:
            # Handle the case where the container does not exist
            print(f"Container '{name}' does not exist.")
            return False
        except Exception as e:
            # Handle other potential errors
            print(f"Failed to delete container '{name}': {e}")
            return False

    def clean(self) -> List[str]:
        # Define the filter for containers with the specific label
        label_filter = {"label": ["provisioner=surfkit"]}

        # Use the filter to list containers
        containers = self.client.containers.list(filters=label_filter, all=True)

        # Initialize a list to keep track of deleted container names or IDs
        deleted_containers = []

        for container in containers:
            try:
                container_name_or_id = (
                    container.name
                )  # or container.id for container ID
                container.remove(force=True)
                print(f"Deleted container: {container_name_or_id}")
                deleted_containers.append(container_name_or_id)
            except Exception as e:
                print(f"Failed to delete container {container_name_or_id}: {e}")

        return deleted_containers
