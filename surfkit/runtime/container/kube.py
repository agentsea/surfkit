import base64
import json
import os
from typing import Literal, List, Optional, Tuple, Type

import httpx
from google.auth.transport.requests import Request
from google.cloud import container_v1
from google.oauth2 import service_account
from kubernetes import client, config
from kubernetes.client.rest import ApiException
from namesgenerator import get_random_name
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt

from .base import ContainerRuntime


class GKEOpts(BaseModel):
    cluster_name: str
    region: str
    service_account_json: str


class LocalOpts(BaseModel):
    path: Optional[str] = os.getenv("KUBECONFIG", os.path.expanduser("~/.kube/config"))


class ConnectConfig(BaseModel):
    provider: Literal["gke", "local"] = "local"
    namespace: str = "default"
    gke_opts: Optional[GKEOpts] = None
    local_opts: Optional[LocalOpts] = None
    branch: Optional[str] = None


def gke_opts_from_env(
    gke_sa_json=os.getenv("GKE_SA_JSON"),
    cluster_name=os.getenv("CLUSTER_NAME"),
    region=os.getenv("CLUSTER_REGION"),
) -> GKEOpts:
    if not gke_sa_json:
        raise ValueError("GKE_SA_JSON not set")
    if not cluster_name:
        raise ValueError("CLUSTER_NAME not set")
    if not region:
        raise ValueError("CLUSTER_REGION not set")
    return GKEOpts(
        service_account_json=gke_sa_json,
        cluster_name=cluster_name,
        region=region,
    )


class KubernetesRuntime(ContainerRuntime):
    """A container runtime that uses Kubernetes to manage Pods directly"""

    def __init__(self, cfg: ConnectConfig) -> None:
        self.cfg = cfg or ConnectConfig()

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

        self.branch = cfg.branch

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
