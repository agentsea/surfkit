import atexit
import base64
import json
import logging
import os
import signal
import subprocess
import sys
import threading
import traceback
from typing import Dict, Iterator, List, Literal, Optional, Tuple, Type, Union

import httpx


def custom_thread_excepthook(args):
    # Format the traceback
    exc = args.exc_type(args.exc_value)
    exc.__traceback__ = args.exc_traceback

    # Perform your custom handling/logging here
    print("Caught unhandled exception in thread:", args.thread.name)
    traceback.print_exception(args.exc_type, args.exc_value, args.exc_traceback)


# This sets a global default for all threads in this interpreter.
threading.excepthook = custom_thread_excepthook

import kubernetes.stream.ws_client as ws_client
from agentdesk.util import find_open_port
from google.auth.transport.requests import Request
from google.cloud import container_v1
from google.oauth2 import service_account
from kubernetes import client, config
from kubernetes.client.rest import ApiException
from mllm import Router
from namesgenerator import get_random_name
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_fixed

from surfkit.server.models import (
    V1AgentType,
    V1LearnTask,
    V1ResourceLimits,
    V1ResourceRequests,
    V1SolveTask,
)
from surfkit.types import AgentType

from .base import AgentInstance, AgentRuntime, AgentStatus

# Monkey patch errors

original_proxy = ws_client.PortForward._proxy


def my_proxy_override(self, *args, **kwargs):
    try:
        return original_proxy(self, *args, **kwargs)
    except Exception as exc:
        # Log the detail
        import traceback

        print(">>> K8s PortForward _proxy crashed with exception:")
        traceback.print_exc()
        # Potentially re-raise if you want it to be truly “unhandled”
        # raise


ws_client.PortForward._proxy = my_proxy_override

logger = logging.getLogger(__name__)


class GKEOpts(BaseModel):
    cluster_name: str
    region: str
    service_account_json: str


class LocalOpts(BaseModel):
    path: Optional[str] = os.getenv("KUBECONFIG", os.path.expanduser("~/.kube/config"))


class KubeConnectConfig(BaseModel):
    provider: Literal["gke", "local"] = "local"
    namespace: str = "default"
    gke_opts: Optional[GKEOpts] = None
    local_opts: Optional[LocalOpts] = None
    branch: Optional[str] = None


class KubeAgentRuntime(AgentRuntime["KubeAgentRuntime", KubeConnectConfig]):
    """A container runtime that uses Kubernetes to manage Pods directly"""

    def __init__(self, cfg: Optional[KubeConnectConfig] = None) -> None:
        self.cfg = cfg or KubeConnectConfig()

        self.kubeconfig = None
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
                self.kubeconfig = opts.path
        else:
            raise ValueError("Unsupported provider: " + cfg.provider)

        self.core_api = client.CoreV1Api()

        self.namespace = cfg.namespace

        self.subprocesses = []
        self.setup_signal_handlers()

        self.branch = cfg.branch

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
        logger.debug("creating secret with envs: ", env_vars)
        secret = client.V1Secret(
            api_version="v1",
            kind="Secret",
            metadata=client.V1ObjectMeta(
                name=name,
                namespace=self.namespace,
                # This ensures that the secret is deleted when the pod is deleted.
                labels={"provisioner": "surfkit"},
            ),
            string_data=env_vars,
            type="Opaque",
        )
        try:
            self.core_api.create_namespaced_secret(
                namespace=self.namespace, body=secret
            )
            print(f"Secret created '{name}'")
            return secret
        except ApiException as e:
            print(f"Failed to create secret: {e}")
            raise

    def create(
        self,
        image: str,
        type: AgentType,
        name: Optional[str] = None,
        resource_requests: V1ResourceRequests = V1ResourceRequests(),
        resource_limits: V1ResourceLimits = V1ResourceLimits(),
        env_vars: Optional[dict] = None,
        owner_id: Optional[str] = None,
        auth_enabled: bool = True,
        labels: Optional[Dict[str, str]] = None,
        check_http_health: bool = True,
    ) -> None:
        if not name:
            name = get_random_name("-")
            if not name:
                raise ValueError("Could not generate a random name")

        if env_vars is None:
            env_vars = {}

        if not auth_enabled:
            env_vars["AGENT_NO_AUTH"] = "true"

        secret = None
        if env_vars:
            # Create a secret for the environment variables
            print("creating secret...")
            secret: Optional[client.V1Secret] = self.create_secret(name, env_vars)
            env_from = [
                client.V1EnvFromSource(
                    secret_ref=client.V1SecretEnvSource(name=secret.metadata.name)  # type: ignore
                )
            ]
        else:
            env_from = []

        # Resource configurations as before
        resources = client.V1ResourceRequirements(
            requests={"memory": resource_requests.memory, "cpu": resource_requests.cpu},
            limits={"memory": resource_limits.memory, "cpu": resource_limits.cpu},
        )
        if resource_requests.gpu:
            raise ValueError("GPU resource requests are not supported")

        logger.debug("using resources: ", resources.__dict__)

        # Container configuration
        container = client.V1Container(
            name=name,
            image=image,
            ports=[client.V1ContainerPort(container_port=9090)],
            resources=resources,
            env_from=env_from,  # Using envFrom to source env vars from the secret
            image_pull_policy="Always",
        )

        # print("\ncreating container: ", container.__dict__)

        # Pod specification
        pod_spec = client.V1PodSpec(
            containers=[container],
            restart_policy="Never",
        )

        # Pod creation
        pod = client.V1Pod(
            api_version="v1",
            kind="Pod",
            metadata=client.V1ObjectMeta(
                name=name,
                labels={"provisioner": "surfkit", **(labels or {})},
                annotations={
                    "owner": owner_id,
                    "agent_name": name,
                    "agent_type": type.name,
                    "agent_model": type.to_v1().model_dump_json(),
                },
            ),
            spec=pod_spec,
        )

        try:
            created_pod: client.V1Pod = self.core_api.create_namespaced_pod(  # type: ignore
                namespace=self.namespace, body=pod
            )
            print(f"Pod created with name '{name}'")
            # print("created pod: ", created_pod.__dict__)
            # Update secret's owner reference UID to newly created pod's UID
            if secret:
                print("updating secret refs...")
                if not secret.metadata:
                    raise ValueError("expected secret metadata to be set")
                if not created_pod.metadata:
                    raise ValueError("expected pod metadata to be set")
                secret.metadata.owner_references = [
                    client.V1OwnerReference(
                        api_version="v1",
                        kind="Pod",
                        name=name,
                        uid=created_pod.metadata.uid,  # This should be set dynamically after pod creation
                    )
                ]
                self.core_api.patch_namespaced_secret(
                    name=secret.metadata.name,
                    namespace=self.namespace,
                    body=secret,  # type: ignore
                )
                print("secret refs updated")
        except ApiException as e:
            print(f"Exception when creating pod: {e}")
            raise

        self.wait_pod_ready(name)
        if check_http_health:
            self.wait_for_http_200(name)

    @classmethod
    def connect_config_type(cls) -> Type[KubeConnectConfig]:
        return KubeConnectConfig

    def connect_config(self) -> KubeConnectConfig:
        return self.cfg

    @classmethod
    def connect(cls, cfg: KubeConnectConfig) -> "KubeAgentRuntime":
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

        logger.debug("K8s getting cluster...")
        cluster_request = container_v1.GetClusterRequest(
            name=f"projects/{project_id}/locations/{opts.region}/clusters/{opts.cluster_name}"
        )
        cluster = gke_service.get_cluster(request=cluster_request)

        # Configure Kubernetes client
        logger.debug("K8s getting token...")
        ca_cert = base64.b64decode(cluster.master_auth.cluster_ca_certificate)
        try:
            logger.debug("K8s refreshing token...")
            credentials.refresh(Request())
        except Exception as e:
            logger.error("K8s token refresh failed: ", e)
            raise e
        access_token = credentials.token
        logger.debug("K8s got token: ", access_token)

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
        logger.debug("K8s returning client...")

        return v1_client, project_id, cluster_name

    @retry(stop=stop_after_attempt(200), wait=wait_fixed(2))
    def wait_for_http_200(self, name: str, path: str = "/", port: int = 9090):
        """
        Waits for an HTTP 200 response from the specified path on the given pod.

        Parameters:
            name (str): The name of the pod.
            path (str): The path to query. Defaults to root '/'.
            port (int): The port on which the pod service is exposed. Defaults to 9090.

        Raises:
            RuntimeError: If the response is not 200 after the specified retries.
        """
        logger.debug(
            f"Checking HTTP 200 readiness for pod {name} on path {path} and port: {port}"
        )
        print("Waiting for agent to be ready...")
        status_code, response_text = self.call(
            name=name, path=path, method="GET", port=port
        )
        if status_code != 200:
            logger.debug(f"Received status code {status_code}, retrying...")
            raise Exception(
                f"Pod {name} at path {path} is not ready. Status code: {status_code}"
            )
        logger.debug(f"Pod {name} at path {path} responded with: {response_text}")
        logger.debug(f"Pod {name} at path {path} is ready with status 200.")
        print(f"Health check passed for agent '{name}'")

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
                        return True
            print("Waiting for pod to be ready...")
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
        port: int = 9090,
        data: Optional[dict] = None,
        headers: Optional[dict] = None,
    ) -> Tuple[int, str]:
        data = data or {}
        headers = headers or {}

        workload_proxy_url = os.getenv("WORKLOAD_PROXY_URL")
        if workload_proxy_url is not None:
            print("Using workload proxy at", workload_proxy_url)
            client_cert = os.getenv("WORKLOAD_PROXY_CLIENT_CERT")
            client_key = os.getenv("WORKLOAD_PROXY_CLIENT_KEY")
            ca_cert = os.getenv("WORKLOAD_PROXY_CA_CERT")

            workload_proxy_client = httpx.Client(
                verify=ca_cert, cert=(client_cert, client_key)
            )

            merged_headers = {
                **headers,
                "X-Pod-Name": name,
                "X-Namespace": self.cfg.namespace,
                "X-Port": str(port),
            }
        else:
            print("Using direct connection to workload service")
            workload_proxy_client = httpx.Client()
            merged_headers = headers
            workload_proxy_url = (
                f"http://{name}.{self.cfg.namespace}.svc.cluster.local:{port}"
            )

        json_data = None if method == "GET" else data
        query_parameters = ""
        if method == "GET" and data:
            query_parameters = "?" + "&".join([f"{k}={v}" for k, v in data.items()])

        url = f"{workload_proxy_url.rstrip('/')}/{path.lstrip('/')}" + query_parameters

        print("Method: ", method)
        print("URL: ", url)
        print("Headers: ", merged_headers)
        print("JSON Data: ", json_data)

        r = workload_proxy_client.request(
            method=method, url=url, headers=merged_headers, json=json_data
        )

        return r.status_code, r.text

    def setup_signal_handlers(self):
        signal.signal(signal.SIGINT, self.graceful_exit)
        signal.signal(signal.SIGTERM, self.graceful_exit)
        atexit.register(self.cleanup_subprocesses)

    def _register_cleanup(self, proc: subprocess.Popen):
        self.subprocesses.append(proc)

    def cleanup_subprocesses(self):
        for proc in self.subprocesses:
            if proc.poll() is None:  # Process is still running
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
        self.subprocesses = []  # Clear the list after cleaning up

    def graceful_exit(self, signum, frame):
        self.cleanup_subprocesses()
        sys.exit(signum)  # Exit with the signal number

    def requires_proxy(self) -> bool:
        """Whether this runtime requires a proxy to be used"""
        return True

    def proxy(
        self,
        name: str,
        local_port: Optional[int] = None,
        agent_port: int = 9090,
        background: bool = True,
        owner_id: Optional[str] = None,
    ) -> Optional[int]:
        """Proxy the agent port to localhost.

        If the runtime is GKE, we load a temporary kubeconfig (the same way
        we do in `connect_to_gke`) so that kubectl uses proper credentials.

        Args:
            name (str): Name of the agent
            local_port (Optional[int], optional): Local port to proxy to. Defaults to None.
            agent_port (int, optional): Agent port. Defaults to 9090.
            background (bool, optional): Whether to run in the background. Defaults to True.
            owner_id (Optional[str], optional): Owner ID. Defaults to None.

        Returns:
            Optional[int]: An optional PID of the proxy when background==True, else None.
        """
        if local_port is None:
            local_port = find_open_port(9090, 10090)

        # Default: no special KUBECONFIG prefix
        kube_cmd_prefix = ""

        # If we're using GKE, generate a temp kube config so kubectl has credentials
        if self.cfg.provider == "gke" and self.cfg.gke_opts:
            import base64
            import tempfile

            import yaml

            # We'll re-gather the GKE config using the same logic as connect_to_gke,
            # but only to build the ephemeral kubeconfig file for the CLI call.
            service_account_info = json.loads(self.cfg.gke_opts.service_account_json)
            credentials = service_account.Credentials.from_service_account_info(
                service_account_info,
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )

            gke_service = container_v1.ClusterManagerClient(credentials=credentials)
            project_id = service_account_info.get("project_id")
            if (
                not project_id
                or not self.cfg.gke_opts.cluster_name
                or not self.cfg.gke_opts.region
            ):
                raise ValueError(
                    "Missing project_id, cluster_name, or region in credentials or metadata"
                )
            cluster_request = container_v1.GetClusterRequest(
                name=(
                    f"projects/{project_id}/locations/"
                    f"{self.cfg.gke_opts.region}/clusters/"
                    f"{self.cfg.gke_opts.cluster_name}"
                )
            )
            cluster = gke_service.get_cluster(request=cluster_request)
            ca_cert = base64.b64decode(cluster.master_auth.cluster_ca_certificate)
            credentials.refresh(Request())
            access_token = credentials.token
            cluster_name = self.cfg.gke_opts.cluster_name

            # Build an ephemeral kubeconfig dict
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

            # Write out the ephemeral kubeconfig
            with tempfile.NamedTemporaryFile("w", delete=False) as tmp:
                yaml.dump(kubeconfig, tmp)
                tmp.flush()
                kube_cmd_prefix = f"KUBECONFIG={tmp.name} "

        # Build port-forward command
        cmd = f"{kube_cmd_prefix}kubectl port-forward pod/{name} {local_port}:{agent_port} -n {self.namespace}"

        if background:
            proc = subprocess.Popen(
                cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            self._register_cleanup(proc)
            return proc.pid  # Return the PID of the subprocess
        else:
            try:
                subprocess.run(cmd, shell=True, check=True)
                return None  # No PID to return when not in background mode
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"Port forwarding failed: {e}")

    def logs(
        self,
        name: str,
        follow: bool = False,
        owner_id: Optional[str] = None,
    ) -> Union[str, Iterator[str]]:
        """
        Fetches the logs from the specified pod. Can return all logs as a single string,
        or stream the logs as a generator of strings.

        Parameters:
            name (str): The name of the pod.
            follow (bool): Whether to continuously follow the logs.
            owner_id (Optional[str]): The owner ID of the pod. If provided, it will be included in the log lines.

        Returns:
            Union[str, Iterator[str]]: All logs as a single string, or a generator that yields log lines.
        """
        try:
            return self.core_api.read_namespaced_pod_log(
                name=name,
                namespace=self.namespace,
                follow=follow,
                pretty="true",
                _preload_content=False,  # Important to return a generator when following
            )
        except ApiException as e:
            print(f"Failed to get logs for pod '{name}': {e}")
            raise

    def list(
        self,
        owner_id: Optional[str] = None,
        source: bool = False,
    ) -> List[AgentInstance]:
        instances = []

        if source:
            try:
                pods = self.core_api.list_namespaced_pod(
                    namespace=self.namespace, label_selector="provisioner=surfkit"
                )
                for pod in pods.items:
                    agent_type_model = pod.metadata.annotations.get("agent_model")
                    if not agent_type_model:
                        continue  # Skip if no agent model annotation

                    agent_type = AgentType.from_v1(
                        V1AgentType.model_validate_json(agent_type_model)
                    )
                    name = pod.metadata.name

                    instances.append(
                        AgentInstance(
                            name,
                            agent_type,
                            self,
                            status=AgentStatus.RUNNING,
                            port=9090,
                        )
                    )
            except ApiException as e:
                print(f"Failed to list pods: {e}")
                raise

        else:
            instances = AgentInstance.find(owner_id=owner_id, runtime_name=self.name())

        return instances

    def get(
        self,
        name: str,
        owner_id: Optional[str] = None,
        source: bool = False,
    ) -> AgentInstance:
        if source:
            try:
                pod = self.core_api.read_namespaced_pod(
                    name=name, namespace=self.namespace
                )
                agent_type_model = pod.metadata.annotations.get("agent_model")  # type: ignore
                if not agent_type_model:
                    raise ValueError("Agent model annotation missing in pod metadata")

                agent_type = AgentType.from_v1(
                    V1AgentType.model_validate_json(agent_type_model)
                )

                return AgentInstance(
                    name, agent_type, self, status=AgentStatus.RUNNING, port=9090
                )
            except ApiException as e:
                print(f"Failed to get pod '{name}': {e}")
                raise

        else:
            instances = AgentInstance.find(
                name=name, owner_id=owner_id, runtime_name=self.name()
            )
            if not instances:
                raise ValueError(f"No agent instance found with name '{name}'")
            return instances[0]

    def exists(
        self,
        name: str,
    ) -> bool:
        try:
            pod = self.core_api.read_namespaced_pod(name=name, namespace=self.namespace)

            agent_type_model = pod.metadata.annotations.get("agent_model")  # type: ignore
            if not agent_type_model:
                raise ValueError("Agent model annotation missing in pod metadata")

            return True
        except ApiException as e:
            print(f"Failed to get pod '{name}': {e}")
            return False

    def delete(
        self,
        name: str,
        owner_id: Optional[str] = None,
    ) -> None:
        try:
            # Delete the pod
            self.core_api.delete_namespaced_pod(
                name=name,
                namespace=self.namespace,
                body=client.V1DeleteOptions(grace_period_seconds=5),
            )
            self.core_api.delete_namespaced_secret(name=name, namespace=self.namespace)
            print(f"Successfully deleted pod: {name}")
        except ApiException as e:
            print(f"Failed to delete pod '{name}': {e}")
            raise

    def runtime_local_addr(self, name: str, owner_id: Optional[str] = None) -> str:
        """
        Returns the local address of the agent with respect to the runtime
        """
        instances = AgentInstance.find(name=name, owner_id=owner_id)
        if not instances:
            raise ValueError(f"No instances found for name '{name}'")
        instance = instances[0]

        return (
            f"http://{instance.name}.{self.namespace}.svc.cluster.local:{instance.port}"
        )

    def clean(
        self,
        owner_id: Optional[str] = None,
    ) -> None:
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
        owner_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        labels: Optional[Dict[str, str]] = None,
        auth_enabled: bool = True,
        debug: bool = False,
        check_http_health: bool = True,
    ) -> AgentInstance:
        logger.debug("creating agent with type: ", agent_type.__dict__)
        if not agent_type.versions:
            raise ValueError("No versions specified in agent type")
        if not version:
            version = list(agent_type.versions.keys())[0]
        img = agent_type.versions.get(version)
        if not img:
            raise ValueError("img not found")
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

        if agent_type.llm_providers:
            env_vars["MODEL_PREFERENCE"] = ",".join(agent_type.llm_providers.preference)

        self.create(
            image=img,
            type=agent_type,
            name=name,
            resource_requests=agent_type.resource_requests,
            resource_limits=agent_type.resource_limits,
            env_vars=env_vars,
            owner_id=owner_id,
            auth_enabled=auth_enabled,
            labels=labels,
            check_http_health=check_http_health,
        )

        return AgentInstance(
            name=name,
            type=agent_type,
            runtime=self,
            status=AgentStatus.RUNNING,
            port=9090,
            version=version,
            owner_id=owner_id,
            tags=tags if tags else [],
            labels=labels if labels else {},
        )

    def _get_headers_with_auth(self, token: Optional[str] = None) -> dict:
        """Helper to return headers with optional Authorization"""
        if not token:
            return {}
        headers = {"Authorization": f"Bearer {token}"}
        return headers

    def learn_task(
        self,
        name: str,
        learn_task: V1LearnTask,
        follow_logs: bool = False,
        attach: bool = False,
    ) -> None:
        try:
            status_code, response_text = self.call(
                name=name,
                path="/v1/learn",
                method="POST",
                port=9090,
                data=learn_task.model_dump(),
                headers=self._get_headers_with_auth(learn_task.task.auth_token),
            )
            logger.debug(f"Skill posted with response: {status_code}, {response_text}")

            if follow_logs:
                print(f"Following logs for '{name}':")
                self._handle_logs_with_attach(name, attach)

        except ApiException as e:
            logger.error(f"API exception occurred: {e}")
            raise
        except Exception as e:
            logger.error(f"An error occurred while posting the task: {e}")
            raise

    def solve_task(
        self,
        name: str,
        task: V1SolveTask,
        follow_logs: bool = False,
        attach: bool = False,
        owner_id: Optional[str] = None,
    ) -> None:
        try:
            status_code, response_text = self.call(
                name=name,
                path="/v1/tasks",
                method="POST",
                port=9090,
                data=task.model_dump(),
                headers=self._get_headers_with_auth(task.task.auth_token),
            )
            logger.debug(f"Task posted with response: {status_code}, {response_text}")

            if follow_logs:
                print(f"Following logs for '{name}':")
                self._handle_logs_with_attach(name, attach)

        except ApiException as e:
            logger.error(f"API exception occurred: {e}")
            raise
        except Exception as e:
            logger.error(f"An error occurred while posting the task: {e}")
            raise

    def _handle_logs_with_attach(self, agent_name: str, attach: bool):
        import typer

        print("following logs for agent...")
        try:
            log_lines = self.logs(name=agent_name, follow=True)

            for line in log_lines:
                clean_line = line.decode("utf-8").strip()  # type: ignore
                print(clean_line)
                if clean_line.startswith("► task run ended"):
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
            print(f"Interrupt received, stopping logs and deleting pod '{agent_name}'")

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
        except ApiException as e:
            print(f"Failed to follow logs for pod '{agent_name}': {e}")
            raise
        except Exception as e:
            print(f"An error occurred while fetching logs: {e}")
            raise

    def refresh(self, owner_id: Optional[str] = None) -> None:
        """
        Synchronizes the state between running Kubernetes pods and the database.
        Ensures that the pods and the database reflect the same set of running agent instances.

        Parameters:
            owner_id (Optional[str]): The ID of the owner to filter instances.
        """
        # Fetch the running pods from Kubernetes
        label_selector = "provisioner=surfkit"
        running_pods = self.core_api.list_namespaced_pod(
            namespace=self.namespace, label_selector=label_selector
        ).items

        # Fetch the agent instances from the database
        db_instances = AgentInstance.find(owner_id=owner_id, runtime_name=self.name())

        # Create a mapping of pod names to pods
        running_pods_map = {pod.metadata.name: pod for pod in running_pods}  # type: ignore

        # Create a mapping of instance names to instances
        db_instances_map = {instance.name: instance for instance in db_instances}

        # Check for pods that are running but not in the database
        for pod_name, pod in running_pods_map.items():
            if pod_name not in db_instances_map:
                print(
                    f"Pod '{pod_name}' is running but not in the database. Creating new instance."
                )
                agent_type_model = pod.metadata.annotations.get("agent_model")
                if not agent_type_model:
                    print(
                        f"Skipping pod '{pod_name}' as it lacks 'agent_model' annotation."
                    )
                    continue

                agent_type = AgentType.from_v1(
                    V1AgentType.model_validate_json(agent_type_model)
                )
                new_instance = AgentInstance(
                    name=pod_name,
                    type=agent_type,
                    runtime=self,
                    status=AgentStatus.RUNNING,
                    port=9090,
                    owner_id=owner_id,
                )
                new_instance.save()

        # Check for instances in the database that are not running as pods
        for instance_name, instance in db_instances_map.items():
            if instance_name not in running_pods_map:
                print(
                    f"Instance '{instance_name}' is in the database but not running. Removing from database."
                )
                instance.delete(force=True)

        logger.debug(
            "Refresh complete. State synchronized between Kubernetes and the database."
        )

    def learn_task_with_job(
        self,
        name: str,
        agent_type: AgentType,
        learn_task: V1LearnTask,
        api_key: str,
        version: Optional[str] = None,
        env_vars: Optional[dict] = None,
        owner_id: Optional[str] = None,
        debug: bool = False,
    ) -> None:
        """
        Creates and launches a Kubernetes Job for a learning task.
        This is very similar to 'run', but instead of creating a Pod,
        it creates a Job resource and injects the V1LearnTask as an env var.

        Args:
            name (str): Name for the job
            agent_type (AgentType): The type describing the agent environment/image
            learn_task (V1LearnTask): The V1LearnTask containing the learning request
            api_key (str): The API key to use for the job
            version (Optional[str], optional): Optional agent version (image tag) to use. Defaults to None.
            env_vars (Optional[dict], optional): Additional environment variables to pass along. Defaults to None.
            owner_id (Optional[str], optional): Owner ID used for labeling/annotations if desired. Defaults to None.
            debug (bool, optional): Whether to enable debug mode. Defaults to False.
        """
        logger.debug(
            "creating job for learning task with agent type: %s", agent_type.__dict__
        )
        if not agent_type.versions:
            raise ValueError("No versions specified in agent type")

        if not version:
            version = list(agent_type.versions.keys())[0]

        image = agent_type.versions.get(version)
        if not image:
            raise ValueError(
                f"No image found for version '{version}' in agent type '{agent_type.name}'"
            )

        # Initialize env_vars if None
        if env_vars is None:
            env_vars = {}

        # Store the learn_task as a JSON-encoded environment variable
        # so the container can parse/use that data at startup
        env_vars["LEARN_TASK_JSON"] = learn_task.model_dump_json()
        env_vars["AGENTSEA_API_KEY"] = api_key

        if debug:
            env_vars["DEBUG"] = "true"

        # Prepare resource requests/limits
        resource_requests = agent_type.resource_requests
        resource_limits = agent_type.resource_limits

        if resource_requests.gpu:
            raise ValueError(
                "GPU resource requests are not supported via this job method yet"
            )

        # Build up the container environment
        # This is simpler than the approach with a Secret, but if you need
        # sensitive data, consider a Secret instead.
        container_env = [
            client.V1EnvVar(name=k, value=str(v)) for k, v in env_vars.items()
        ]

        resources = client.V1ResourceRequirements(
            requests={"memory": resource_requests.memory, "cpu": resource_requests.cpu},
            limits={"memory": resource_limits.memory, "cpu": resource_limits.cpu},
        )
        logger.debug("Job will use resources: %s", resources.__dict__)

        # Create the container spec
        container = client.V1Container(
            name=name,
            command=["poetry", "run", "python", "-m", "foo.learner"],
            image=image,
            image_pull_policy="Always",
            env=container_env,
            resources=resources,
        )

        # Build the Pod spec (template for the Job)
        pod_spec = client.V1PodSpec(
            containers=[container],
            restart_policy="Never",
        )

        # Incorporate any relevant annotations/labels
        annotations = {
            "owner": owner_id if owner_id else "",
            "agent_name": name,
            "agent_type": agent_type.name,
            "agent_model": agent_type.to_v1().model_dump_json(),
        }
        labels = {"provisioner": "surfkit"}

        # Construct the pod template
        template = client.V1PodTemplateSpec(
            metadata=client.V1ObjectMeta(
                labels=labels,
                annotations=annotations,
            ),
            spec=pod_spec,
        )

        # Construct the Job spec
        job_spec = client.V1JobSpec(
            template=template,
            backoff_limit=0,  # e.g. for tasks that should run exactly once
            ttl_seconds_after_finished=60 * 60 * 24,  # 1 day
        )

        # Construct the Job object
        job = client.V1Job(
            api_version="batch/v1",
            kind="Job",
            metadata=client.V1ObjectMeta(
                name=name,
                labels=labels,
            ),
            spec=job_spec,
        )

        # Create the Job in the namespace
        batch_api = client.BatchV1Api()
        try:
            batch_api.create_namespaced_job(namespace=self.namespace, body=job)
            logger.info(f"Job '{name}' created successfully for learn task.")
            print(f"Job created: {name}")
        except ApiException as e:
            logger.error(f"Failed to create job '{name}' for learn task: {e}")
            raise
