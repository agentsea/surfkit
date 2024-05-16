import base64
import json
import os
import socket
import urllib.error
import urllib.parse
import urllib.request
from typing import List, Optional, Tuple, Type

from google.auth.transport.requests import Request
from google.cloud import container_v1
from google.oauth2 import service_account
from kubernetes import client, config, stream
from kubernetes.client import Configuration
from kubernetes.client.api import core_v1_api
from kubernetes.client.rest import ApiException
from kubernetes.stream import portforward
from namesgenerator import get_random_name
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt

from .base import ContainerRuntime


class GKEOpts(BaseModel):
    cluster_name: str
    region: str
    service_account_json: str


class LocalOpts(BaseModel):
    path: Optional[str] = os.getenv("KUBECONFIG")


class ConnectConfig(BaseModel):
    provider: str = "local"
    namespace: str = "default"
    gke_opts: Optional[GKEOpts] = None
    local_opts: Optional[LocalOpts] = None


class KubernetesRuntime(ContainerRuntime):
    """A container runtime that uses Kubernetes to manage Pods directly"""

    def __init__(self, cfg: ConnectConfig) -> None:
        # Load the Kubernetes configuration, typically from ~/.kube/config
        if cfg.provider == "gke":
            opts = cfg.gke_opts
            if not opts:
                raise ValueError("GKE opts missing")
            self.connect_to_gke(opts)
        elif cfg.provider == "local":
            opts = cfg.local_opts
            if not opts:
                raise ValueError("Local opts missing")
            if opts.path:
                config.load_kube_config(opts.path)
        else:
            raise ValueError("Unsupported provider: " + cfg.provider)

        self.core_api = core_v1_api.CoreV1Api()
        self.namespace = cfg.namespace

    @classmethod
    def name(cls) -> str:
        return "kube"

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

        labels = {"provisioner": "surfkit"}

        # Define resource requirements
        resources = client.V1ResourceRequirements(
            requests={"memory": mem_request, "cpu": cpu_request},
            limits={"memory": mem_limit, "cpu": cpu_limit},
        )

        if gpu_mem:
            if "limits" in resources:  # type: ignore
                resources.limits["nvidia.com/gpu"] = gpu_mem  # type: ignore
            else:
                resources.limits = {"nvidia.com/gpu": gpu_mem}

        # Define the container with environment variables
        env_list = []
        if env_vars:
            for key, value in env_vars.items():
                env_list.append(client.V1EnvVar(name=key, value=value))

        container = client.V1Container(
            name=name,
            image=image,
            ports=[client.V1ContainerPort(container_port=8080)],
            resources=resources,
            env=env_list,
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

    @classmethod
    def connect_config_type(cls) -> Type[ConnectConfig]:
        return ConnectConfig

    @classmethod
    def connect(cls, cfg: ConnectConfig) -> "KubernetesRuntime":
        return cls(cfg)

    @retry(stop=stop_after_attempt(15))
    def connect_to_gke(self, opts: GKEOpts) -> Tuple[client.CoreV1Api, str, str]:
        """
        Sets up and returns a configured Kubernetes client (CoreV1Api) and cluster details.

        Returns:
            Tuple containing the Kubernetes CoreV1Api client object, the project ID, and the cluster name.
        """
        service_account_info = json.loads(opts.service_account_json)
        credentials = service_account.Credentials.from_service_account_info(
            service_account_info,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )

        # Setup GKE client to get cluster information
        gke_service = container_v1.ClusterManagerClient(credentials=credentials)
        project_id = service_account_info.get("project_id")
        if not project_id or not opts.cluster_name or not opts.region:
            raise ValueError(
                "Missing project_id, cluster_name, or region in credentials or metadata"
            )

        print("\nK8s getting cluster...")
        cluster_request = container_v1.GetClusterRequest(
            name=f"projects/{project_id}/locations/{opts.region}/clusters/{opts.cluster_name}"
        )
        cluster = gke_service.get_cluster(request=cluster_request)

        # Configure Kubernetes client
        print("\nK8s getting token...")
        ca_cert = base64.b64decode(cluster.master_auth.cluster_ca_certificate)
        try:
            print("\nK8s refreshing token...")
            credentials.refresh(Request())
        except Exception as e:
            print("\nK8s token refresh failed: ", e)
            raise e
        access_token = credentials.token
        print("\nK8s got token: ", access_token)

        cluster_name = opts.cluster_name

        kubeconfig = {
            "apiVersion": "v1",
            "kind": "Config",
            "clusters": [
                {
                    "name": cluster_name,
                    "cluster": {
                        "server": f"https://{cluster.endpoint}",
                        "certificate-authority-data": base64.b64encode(
                            ca_cert
                        ).decode(),
                    },
                }
            ],
            "contexts": [
                {
                    "name": cluster_name,
                    "context": {
                        "cluster": cluster_name,
                        "user": cluster_name,
                    },
                }
            ],
            "current-context": cluster_name,
            "users": [
                {
                    "name": cluster_name,
                    "user": {
                        "token": access_token,
                    },
                }
            ],
        }

        config.load_kube_config_from_dict(config_dict=kubeconfig)
        v1_client = client.CoreV1Api()
        print("\nK8s returning client...")

        return v1_client, project_id, cluster_name

    @retry(stop=stop_after_attempt(15))
    def call(
        self,
        name: str,
        path: str,
        method: str,
        port: int = 8080,
        data: Optional[dict] = None,
        headers: Optional[dict] = None,
    ) -> Tuple[int, str]:

        c = Configuration.get_default_copy()
        c.assert_hostname = False  # type: ignore
        Configuration.set_default(c)
        core_v1 = client.CoreV1Api()
        ##############################################################################
        # Kubernetes pod port forwarding works by directly providing a socket which
        # the python application uses to send and receive data on. This is in contrast
        # to the go client, which opens a local port that the go application then has
        # to open to get a socket to transmit data.
        #
        # This simplifies the python application, there is not a local port to worry
        # about if that port number is available. Nor does the python application have
        # to then deal with opening this local port. The socket used to transmit data
        # is immediately provided to the python application.
        #
        # Below also is an example of monkey patching the socket.create_connection
        # function so that DNS names of the following formats will access kubernetes
        # ports:
        #
        #    <pod-name>.<namespace>.kubernetes
        #    <pod-name>.pod.<namespace>.kubernetes
        #    <service-name>.svc.<namespace>.kubernetes
        #    <service-name>.service.<namespace>.kubernetes
        #
        # These DNS name can be used to interact with pod ports using python libraries,
        # such as urllib.request and http.client. For example:
        #
        # response = urllib.request.urlopen(
        #     'https://metrics-server.service.kube-system.kubernetes/'
        # )
        #
        ##############################################################################

        # Monkey patch socket.create_connection which is used by http.client and
        # urllib.request. The same can be done with urllib3.util.connection.create_connection
        # if the "requests" package is used.
        socket_create_connection = socket.create_connection

        def kubernetes_create_connection(address, *args, **kwargs):
            dns_name = address[0]
            if isinstance(dns_name, bytes):
                dns_name = dns_name.decode()
            dns_name = dns_name.split(".")
            if dns_name[-1] != "kubernetes":
                return socket_create_connection(address, *args, **kwargs)
            if len(dns_name) not in (3, 4):
                raise RuntimeError("Unexpected kubernetes DNS name.")
            namespace = dns_name[-2]
            name = dns_name[0]
            port = address[1]
            print("connecting to: ", namespace, name, port)
            if len(dns_name) == 4:
                if dns_name[1] in ("svc", "service"):
                    service = core_v1.read_namespaced_service(name, namespace)
                    for service_port in service.spec.ports:  # type: ignore
                        if service_port.port == port:
                            port = service_port.target_port
                            break
                    else:
                        raise RuntimeError(f"Unable to find service port: {port}")
                    label_selector = []
                    for key, value in service.spec.selector.items():  # type: ignore
                        label_selector.append(f"{key}={value}")
                    pods = core_v1.list_namespaced_pod(
                        namespace, label_selector=",".join(label_selector)
                    )
                    if not pods.items:
                        raise RuntimeError("Unable to find service pods.")
                    name = pods.items[0].metadata.name
                    if isinstance(port, str):
                        for container in pods.items[0].spec.containers:
                            for container_port in container.ports:
                                if container_port.name == port:
                                    port = container_port.container_port
                                    break
                            else:
                                continue
                            break
                        else:
                            raise RuntimeError(
                                f"Unable to find service port name: {port}"
                            )
                elif dns_name[1] != "pod":
                    raise RuntimeError(f"Unsupported resource type: {dns_name[1]}")
            pf = portforward(
                core_v1.connect_get_namespaced_pod_portforward,
                name,
                namespace,
                ports=str(port),
            )
            return pf.socket(port)

        socket.create_connection = kubernetes_create_connection

        namespace = self.namespace
        if not namespace:
            raise ValueError("NAMESPACE environment variable not set")
        # Access the nginx http server using the
        # "<pod-name>.pod.<namespace>.kubernetes" dns name.
        # Construct the URL with the custom path
        url = f"http://{name.lower()}.pod.{namespace}.kubernetes:{port}{path}"

        # Create a request object based on the HTTP method
        if method.upper() == "GET":
            if data:
                # Convert data to URL-encoded query parameters for GET requests
                query_params = urllib.parse.urlencode(data)
                url += f"?{query_params}"
            request = urllib.request.Request(url)
        else:
            # Set the request method and data for POST, PUT, etc.
            request = urllib.request.Request(url, method=method.upper())
            if data:
                # Convert data to JSON string and set the request body
                request.add_header("Content-Type", "application/json")
                if headers:
                    for k, v in headers.items():
                        request.add_header(k, v)
                request.data = json.dumps(data).encode("utf-8")
            print(f"Request Data: {request.data}")

        # Send the request and handle the response
        try:
            response = urllib.request.urlopen(request)
            status_code = response.code
            response_text = response.read().decode("utf-8")
            print(f"Status Code: {status_code}")

            # Parse the JSON response and return a dictionary
            return status_code, response_text
        except urllib.error.HTTPError as e:
            status_code = e.code
            error_message = e.read().decode("utf-8")
            print(f"Error: {status_code}")
            print(error_message)

            raise SystemError(
                f"Error making http request kubernetes pod {status_code}: {error_message}"
            )
        finally:
            try:
                if response:  # type: ignore
                    response.close()
            except:
                pass

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
