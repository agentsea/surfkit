from typing import List, Optional

from kubernetes import client, config
from kubernetes.client.rest import ApiException
from namesgenerator import get_random_name


class KubernetesRuntime:
    """A container runtime that uses Kubernetes to manage Pods directly"""

    def __init__(self) -> None:
        # Load the Kubernetes configuration, typically from ~/.kube/config
        config.load_kube_config()
        self.core_api = client.CoreV1Api()

    def create(self, image: str, name: Optional[str] = None) -> None:
        if not name:
            name = get_random_name("-")

        labels = {"provisioner": "surfkit"}

        # Define the container
        container = client.V1Container(
            name=name, image=image, ports=[client.V1ContainerPort(container_port=8080)]
        )

        # Create a Pod specification
        pod_spec = client.V1PodSpec(
            containers=[container],
            restart_policy="Never",  # 'Always' if you want the pod to restart on failure
        )

        # Create the Pod
        pod = client.V1Pod(
            api_version="v1",
            kind="Pod",
            metadata=client.V1ObjectMeta(name=name, labels=labels),
            spec=pod_spec,
        )

        # Launch the Pod
        print("Creating pod")
        try:
            self.core_api.create_namespaced_pod(namespace="default", body=pod)
            print(f"Pod created. name='{name}'")
        except ApiException as e:
            print(f"Exception when creating pod: {e}")

    def logs(self, name: str) -> str:
        """
        Fetches the logs from the specified pod.

        Parameters:
            name (str): The name of the pod.

        Returns:
            str: The logs from the pod.
        """
        try:
            return self.core_api.read_namespaced_pod_log(name=name, namespace="default")
        except ApiException as e:
            print(f"Failed to get logs for pod '{name}': {e}")
            raise

    def list(self) -> List[str]:
        pods = self.core_api.list_namespaced_pod(
            namespace="default", label_selector="provisioner=surfkit"
        )
        return [pod.metadata.name for pod in pods.items]

    def delete(self, name: str) -> None:
        try:
            # Delete the pod
            self.core_api.delete_namespaced_pod(
                name=name,
                namespace="default",
                body=client.V1DeleteOptions(grace_period_seconds=5),
            )
            print(f"Successfully deleted pod: {name}")
        except ApiException as e:
            print(f"Failed to delete pod '{name}': {e}")
            raise

    def clean(self) -> None:
        pods = self.core_api.list_namespaced_pod(
            namespace="default", label_selector="provisioner=surfkit"
        )
        for pod in pods.items:
            try:
                self.core_api.delete_namespaced_pod(
                    name=pod.metadata.name,
                    namespace="default",
                    body=client.V1DeleteOptions(grace_period_seconds=5),
                )
                print(f"Deleted pod: {pod.metadata.name}")
            except ApiException as e:
                print(f"Failed to delete pod '{pod.metadata.name}': {e}")
