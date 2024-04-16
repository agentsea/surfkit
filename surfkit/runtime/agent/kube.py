from curses.ascii import isdigit
from typing import List, Optional, Tuple, Type, Union, Iterator
import os
import json
import urllib.error
import urllib.parse
import urllib.request
from tenacity import retry, stop_after_attempt, wait_fixed
import socket
import base64
import subprocess
import atexit
import signal

from kubernetes import client, config
from google.oauth2 import service_account
from google.cloud import container_v1
from google.auth.transport.requests import Request
from kubernetes.client.rest import ApiException
from kubernetes.stream import portforward
from kubernetes.client import Configuration
from kubernetes.client.api import core_v1_api
from namesgenerator import get_random_name
from tenacity import retry
from pydantic import BaseModel
from taskara.models import SolveTaskModel
from agentdesk.util import find_open_port

from .base import AgentRuntime, AgentInstance
from surfkit.types import AgentType
from surfkit.llm import LLMProvider


class GKEOpts(BaseModel):
    cluster_name: str
    region: str
    service_account_json: str


class LocalOpts(BaseModel):
    path: Optional[str] = os.getenv("KUBECONFIG", os.path.expanduser("~/.kube/config"))


class ConnectConfig(BaseModel):
    provider: str = "local"
    namespace: str = "default"
    gke_opts: Optional[GKEOpts] = None
    local_opts: Optional[LocalOpts] = None


class KubernetesAgentRuntime(AgentRuntime):
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
                opts = LocalOpts()
            if opts.path:
                config.load_kube_config(opts.path)
        else:
            raise ValueError("Unsupported provider: " + cfg.provider)

        self.core_api = core_v1_api.CoreV1Api()
        self.namespace = cfg.namespace

    @classmethod
    def name(cls) -> str:
        return "kube"

    def create_secret(self, name: str, env_vars: dict) -> client.V1Secret:
        """
        Creates a Kubernetes Secret object to store environment variables.

        Parameters:
            name (str): The base name of the secret, usually related to the pod name.
            env_vars (dict): A dictionary containing the environment variables as key-value pairs.

        Returns:
            client.V1Secret: The created Kubernetes Secret object.
        """
        print("creating secret with envs: ", env_vars)
        secret = client.V1Secret(
            api_version="v1",
            kind="Secret",
            metadata=client.V1ObjectMeta(
                name=name,
                namespace=self.namespace,
                # This ensures that the secret is deleted when the pod is deleted.
                labels={"provisioner": "surfkit"},
                owner_references=[
                    client.V1OwnerReference(
                        api_version="v1",
                        kind="Pod",
                        name=name,
                        uid="the UID of the Pod here",  # This should be set dynamically after pod creation
                    )
                ],
            ),
            string_data=env_vars,
            type="Opaque",
        )
        try:
            self.core_api.create_namespaced_secret(
                namespace=self.namespace, body=secret
            )
            print(f"Secret created: {name}")
            return secret
        except ApiException as e:
            print(f"Failed to create secret: {e}")
            raise

    def create(
        self,
        image: str,
        name: Optional[str] = None,
        mem_request: Optional[str] = "500m",
        mem_limit: Optional[str] = "2Gi",
        cpu_request: Optional[str] = "1",
        cpu_limit: Optional[str] = "4",
        gpu_mem: Optional[str] = None,
        env_vars: Optional[dict] = None,
    ) -> None:
        if not name:
            name = get_random_name("-")
            if not name:
                raise ValueError("Could not generate a random name")

        secret = None
        if env_vars:
            # Create a secret for the environment variables
            print("creating secret...")
            secret: Optional[client.V1Secret] = self.create_secret(name, env_vars)
            print("secret created: ", secret.__dict__)
            env_from = [
                client.V1EnvFromSource(
                    secret_ref=client.V1SecretEnvSource(name=secret.metadata.name)  # type: ignore
                )
            ]
        else:
            env_from = []

        # Resource configurations as before
        resources = client.V1ResourceRequirements(
            requests={"memory": mem_request, "cpu": cpu_request},
            limits={"memory": mem_limit, "cpu": cpu_limit},
        )
        if gpu_mem:
            raise ValueError("GPU support not yet implemented")

        print("\nusing resources: ", resources.__dict__)

        # Container configuration
        container = client.V1Container(
            name=name,
            image=image,
            ports=[client.V1ContainerPort(container_port=8000)],
            resources=resources,
            env_from=env_from,  # Using envFrom to source env vars from the secret
            image_pull_policy="Always",
        )

        print("\ncreating container: ", container.__dict__)

        # Pod specification
        pod_spec = client.V1PodSpec(
            containers=[container],
            restart_policy="Never",
        )

        # Pod creation
        pod = client.V1Pod(
            api_version="v1",
            kind="Pod",
            metadata=client.V1ObjectMeta(name=name, labels={"provisioner": "surfkit"}),
            spec=pod_spec,
        )

        try:
            created_pod = self.core_api.create_namespaced_pod(
                namespace=self.namespace, body=pod
            )
            print(f"Pod created with name='{name}'")
            # Update secret's owner reference UID to newly created pod's UID
            if secret:
                print("updating secret refs...")
                secret.metadata.owner_references[0].uid = created_pod.metadata.uid  # type: ignore
                self.core_api.patch_namespaced_secret(
                    name=secret.metadata.name, namespace=self.namespace, body=secret  # type: ignore
                )
                print("secret refs updated")
        except ApiException as e:
            print(f"Exception when creating pod: {e}")
            raise

        self.wait_pod_ready(name)
        self.wait_for_http_200(name)

    @classmethod
    def connect_config_type(cls) -> Type[ConnectConfig]:
        return ConnectConfig

    @classmethod
    def connect(cls, cfg: ConnectConfig) -> "KubernetesAgentRuntime":
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

    @retry(stop=stop_after_attempt(200), wait=wait_fixed(2))
    def wait_for_http_200(self, name: str, path: str = "/", port: int = 8000):
        """
        Waits for an HTTP 200 response from the specified path on the given pod.

        Parameters:
            name (str): The name of the pod.
            path (str): The path to query. Defaults to root '/'.
            port (int): The port on which the pod service is exposed. Defaults to 8000.

        Raises:
            RuntimeError: If the response is not 200 after the specified retries.
        """
        print(f"Checking HTTP 200 readiness for pod {name} on path {path}")
        status_code, response_text = self.call(name, path, "GET", port)
        if status_code != 200:
            print(f"Received status code {status_code}, retrying...")
            raise Exception(
                f"Pod {name} at path {path} is not ready. Status code: {status_code}"
            )
        print(f"Pod {name} at path {path} is ready with status 200.")

    @retry(stop=stop_after_attempt(200), wait=wait_fixed(2))
    def wait_pod_ready(self, name: str) -> bool:
        """
        Checks if the specified pod is ready to serve requests.

        Parameters:
            name (str): The name of the pod to check.

        Returns:
            bool: True if the pod is ready, False otherwise.
        """
        try:
            pod = self.core_api.read_namespaced_pod(name=name, namespace=self.namespace)
            conditions = pod.status.conditions  # type: ignore
            if conditions:
                for condition in conditions:
                    if condition.type == "Ready" and condition.status == "True":
                        print("pod is ready!")
                        return True
            print("pod is not ready yet...")
            raise Exception(f"Pod {name} is not ready")
        except ApiException as e:
            print(f"Failed to read pod status for '{name}': {e}")
            raise

    @retry(stop=stop_after_attempt(15))
    def call(
        self,
        name: str,
        path: str,
        method: str,
        port: int = 8000,
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
            # print("connecting to: ", namespace, name, port)
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

    def proxy(
        self,
        name: str,
        local_port: Optional[int] = None,
        pod_port: int = 8000,
        background: bool = True,
    ) -> None:
        """
        Sets up a port forwarding between a local port and a pod port using kubectl command.

        Parameters:
            name (str): The name of the pod to port-forward to.
            local_port (int): The local port number to forward to pod_port.
            pod_port (int): The pod port number to be forwarded, defaults to 8000.
            background (bool): If True, runs the port forwarding in the background and returns immediately.
        """
        if local_port is None:
            local_port = find_open_port(8000, 9000)
        cmd = f"kubectl port-forward pod/{name} {local_port}:{pod_port} -n {self.namespace}"
        if background:
            # Start the subprocess in the background
            proc = subprocess.Popen(
                cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            self._register_cleanup(proc)
            print(
                f"Port-forwarding setup: Local port {local_port} -> Pod {name}:{pod_port}"
            )
        else:
            # Run the subprocess in the foreground, block until the user terminates it
            try:
                print(
                    f"Starting port-forwarding: Local port {local_port} -> Pod {name}:{pod_port}"
                )
                subprocess.run(cmd, shell=True, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Failed to start port-forwarding: {e}")
                raise

    def _register_cleanup(self, proc: subprocess.Popen):
        """
        Registers cleanup actions on process termination.

        Parameters:
            proc (subprocess.Popen): The process to clean up.
        """

        def cleanup():
            if proc.poll() is None:  # Process is still running
                proc.terminate()  # or send any other signal you think necessary
                try:
                    proc.wait(timeout=10)  # Give it a few seconds to terminate
                except subprocess.TimeoutExpired:
                    proc.kill()  # Force kill if not terminated in time
                print("Port-forwarding process terminated.")

        atexit.register(cleanup)
        signal.signal(signal.SIGINT, lambda s, f: cleanup())
        signal.signal(signal.SIGTERM, lambda s, f: cleanup())

    def logs(self, name: str, follow: bool = False) -> Union[str, Iterator[str]]:
        """
        Fetches the logs from the specified pod. Can return all logs as a single string,
        or stream the logs as a generator of strings.

        Parameters:
            name (str): The name of the pod.
            follow (bool): Whether to continuously follow the logs.

        Returns:
            Union[str, Iterator[str]]: All logs as a single string, or a generator that yields log lines.
        """
        try:
            return self.core_api.read_namespaced_pod_log(
                name=name,
                namespace=self.namespace,
                follow=follow,
                _preload_content=False,  # Important to return a generator when following
            )
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

    def run(
        self,
        agent_type: AgentType,
        name: str,
        version: Optional[str] = None,
        env_vars: Optional[dict] = None,
        llm_providers_local: bool = False,
    ) -> AgentInstance:
        print("creating agent with type: ", agent_type.__dict__)
        if not agent_type.image:
            raise ValueError(f"Image not specified for agent type: {agent_type.name}")
        img = agent_type.image
        if version:
            if not agent_type.versions:
                raise ValueError("version supplied but no versions in type")
            img = agent_type.versions.get(version)
        if not img:
            raise ValueError("img not found")
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

        self.create(
            image=img,
            name=name,
            mem_request=agent_type.mem_request,
            mem_limit=agent_type.mem_limit,
            cpu_request=agent_type.cpu_request,
            cpu_limit=agent_type.cpu_limit,
            gpu_mem=agent_type.gpu_mem,
            env_vars=env_vars,
        )

        return AgentInstance(name, agent_type, self)

    def solve_task(
        self, agent_name: str, task: SolveTaskModel, follow_logs: bool = False
    ) -> None:
        try:
            # Making the call to the specified path to initiate the task
            status_code, response_text = self.call(
                name=agent_name,
                path="/v1/tasks",
                method="POST",
                port=8000,
                data=task.model_dump(),
            )
            print(
                f"Task initiation response: Status Code {status_code}, Response Text {response_text}"
            )

        except ApiException as e:
            print(f"API exception occurred: {e}")
            raise
        except Exception as e:
            print(f"An error occurred while posting the task: {e}")
            raise

        if follow_logs:
            print(f"Starting to follow logs for: {agent_name}")
            try:
                log_lines = self.logs(name=agent_name, follow=True)
                for line in log_lines:
                    print(line)
            except ApiException as e:
                print(f"Failed to follow logs for pod '{agent_name}': {e}")
                raise
            except Exception as e:
                print(f"An error occurred while fetching logs: {e}")
                raise