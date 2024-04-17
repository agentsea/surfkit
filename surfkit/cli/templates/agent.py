from surfkit.types import AgentType


def generate_dockerfile(package_name: str) -> str:
    return f"""
# Use an official Python runtime as a parent image
FROM python:3.10

# Set the working directory in the container to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY requirements.txt .

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    python3-dev \
    git \
    ntp \
    rm -rf /var/lib/apt/lists/*  # Clean up to reduce image size

# Upgrade pip
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Surfkit
RUN pip install surfkit

COPY . .

# Run the application
CMD ["uvicorn", "{package_name}.server:app", "--host=0.0.0.0", "--port=9090", "--log-level", "debug"]
"""


def generate_server(agent_name: str) -> str:
    return f"""
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import os
from contextlib import asynccontextmanager

from tenacity import retry, stop_after_attempt, wait_fixed
from taskara.models import SolveTaskModel
from taskara.task import Task
from taskara.models import TaskModel, TasksModel
from surfkit.hub import Hub
from surfkit.llm import LLMProvider
from agentdesk import Desktop

from .agent import {agent_name}

HUB_SERVER = os.environ.get("SURF_HUB_SERVER", "https://surf.agentlabs.xyz")
HUB_API_KEY = os.environ.get("HUB_API_KEY")
if not HUB_API_KEY:
    raise Exception("$HUB_API_KEY not set")
AGENTD_ADDR = os.environ.get("AGENTD_ADDR")
if not AGENTD_ADDR:
    raise Exception("$AGENTD_ADDR not set")
AGENTD_PRIVATE_SSH_KEY = os.environ.get("AGENTD_PRIVATE_SSH_KEY")
if not AGENTD_PRIVATE_SSH_KEY:
    raise Exception("$AGENTD_PRIVATE_SSH_KEY not set")


# Get the LLM provider from env
llm_provider = LLMProvider.from_env()


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "Agent in the shell"}


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/v1/tasks")
async def create_task(background_tasks: BackgroundTasks, task: SolveTaskModel):
    print(f"solving task: \n{{task.model_dump()}}")
    try:
        # TODO: we need to find a way to do this earlier but get status back
        llm_provider.check_model()
    except Exception as e:
        print(f"Cannot connect to LLM providers: {{e}} -- did you provide a valid key?")
        return {
            "status": "failed",
            "message": f'failed to conect to LLM providers -- did you provide a valid key?',
        }

    background_tasks.add_task(_create_task, task)
    print("created background task...")


def _create_task(task: SolveTaskModel):
    hub = Hub(HUB_SERVER)
    user_info = hub.get_user_info(HUB_API_KEY)
    print("got user info: ", user_info.__dict__)

    print("syncing to remote task: ", task.task.id, HUB_SERVER)
    rtask = get_remote_task(
        id=task.task.id,
        owner_id=user_info.email,
    )
    print("got remote task: ", rtask.__dict__)

    print("Saving remote tasks status to running...")
    rtask.status = "in progress"
    rtask.save()

    rtask.post_message("assistant", f"Starting task '{{rtask.description}}'")
    print("creating threads...")

    rtask.create_thread("debug")
    rtask.post_message("assistant", f"I'll post debug messages here", thread="debug")

    rtask.create_thread("prompt")
    rtask.post_message(
        "assistant", f"I'll post all llm prompts that take place here", thread="prompt"
    )
    print("created work thread 'debug' and 'prompt'")

    print("creating desktop...")
    desktop: SemanticDesktop = SemanticDesktop(
        task=rtask,
        agentd_url=AGENTD_ADDR,
        requires_proxy=True,
        proxy_port=7000,
        private_ssh_key=AGENTD_PRIVATE_SSH_KEY,
    )
    print("created desktop: ", desktop.__dict__)

    try:
        fin_task = agent.solve_task(rtask, desktop, task.max_steps, task.site)
    except Exception as e:
        print("error running agent: ", e)
        rtask.status = "failed"
        rtask.error = str(e)
        rtask.save()
        rtask.post_message(
            "assistant", f"Failed to run task '{{rtask.description}}': {{e}}"
        )
        raise e
    if fin_task:
        fin_task.save()

@retry(stop=stop_after_attempt(10), wait=wait_fixed(10))
def get_remote_task(id: str, owner_id: str) -> Task:
    print("connecting to remote task: ", id, HUB_SERVER, HUB_API_KEY)
    try:
        tasks = Task.find(
            id=id,
            remote=HUB_SERVER,
            owner_id=owner_id,
        )
        if not tasks:
            raise Exception(f"Task {{id}} not found")
        print("got remote task: ", tasks[0].__dict__)
        return tasks[0]
    except Exception as e:
        print("error getting remote task: ", e)
        raise e

"""


def generate_main() -> str:
    return f"""
import argparse
import logging
import yaml

from rich.console import Console
from agentdesk.device import Desktop, ProvisionConfig
from agentdesk.vm.gce import GCEProvider
from agentdesk.vm.ec2 import EC2Provider
from agentdesk.vm.qemu import QemuProvider
from taskara import Task
from taskara.models import SolveTaskModel
from surfkit.types import AgentType
from surfkit.models import AgentTypeModel
from surfkit.runtime.agent.load import (
    load_agent_runtime,
    AgentRuntimeConfig,
    DockerConnectConfig,
    KubeConnectConfig,
)
from namesgenerator import get_random_name


console = Console()

DEFAULT_PROXY_PORT = 9123

parser = argparse.ArgumentParser(description="Run the agent with optional debug mode.")
parser.add_argument(
    "--debug",
    action="store_true",
    help="Enable debug mode for more verbose output.",
    default=False,
)
parser.add_argument(
    "--task",
    type=str,
    help="Specify the task to run.",
    required=True,
)
parser.add_argument(
    "--max_steps",
    type=int,
    help="Max steps the agent can take",
    default=10,
)
parser.add_argument(
    "--site",
    type=str,
    help="Max steps the agent can take",
    default=None,
)
parser.add_argument(
    "--version",
    type=str,
    help="Agent version to use",
    default=None,
)
parser.add_argument(
    "--device-runtime",
    type=str,
    help="Device runtime to use",
    default="gce",
)
parser.add_argument(
    "--agent-runtime",
    type=str,
    help="Agent runtime to use",
    default="kube",
)
parser.add_argument(
    "--region",
    type=str,
    help="Region to use",
    default="us-east-1",
)
parser.add_argument(
    "--agent-type",
    type=str,
    help="Agent type filepath",
    default="./agent.yaml",
)
parser.add_argument(
    "--name",
    type=str,
    help="Agent name",
    default=get_random_name("-"),
)
parser.add_argument(
    "--namespace",
    type=str,
    help="Kubernetes namespace",
    default="default",
)
args = parser.parse_args()

if args.debug:
    logging.basicConfig(level=logging.DEBUG)
else:
    logging.basicConfig(level=logging.INFO)

console.print(
    f"solving task '{{args.task}}' on site '{{args.site}}' with max steps {{args.max_steps}}",
    style="green",
)

if args.device_runtime == "gce":
    provider = GCEProvider()
elif args.device_runtime == "ec2":
    provider = EC2Provider(region=args.region)
else:
    provider = QemuProvider()

parameters = {{"site": args.site}}
task = Task(description=args.task, owner_id="local", parameters=parameters)

with open(args.agent_type) as f:
    agent_data = yaml.safe_load(f)
    agent_schema = AgentTypeModel(**agent_data)
agent_type = AgentType.from_schema(agent_schema)

console.print("Creating device...", style="green")
print("\nprovider data: ", provider.to_data())
config = ProvisionConfig(provider=provider.to_data())
device: Desktop = Desktop.ensure("gpt4v-demo1", config=config)

# View the desktop, we'll run in the background so it doesn't block
device.view(background=True)

if args.agent_runtime == "docker":
    print("using docker runtime...")
    dconf = DockerConnectConfig()
    conf = AgentRuntimeConfig(provider="docker", docker_config=dconf)
elif args.agent_runtime == "kube":
    print("using kube runtime...")
    kconf = KubeConnectConfig()
    conf = AgentRuntimeConfig(provider="kube", kube_config=kconf)
else:
    raise ValueError("Unknown agent runtime")

runtime = load_agent_runtime(conf)

print("\ndevice schema: ", device.to_schema())
console.print("Running agent...", style="green")
agent = runtime.run(agent_type, args.name, args.version, llm_providers_local=True)

agent.proxy(DEFAULT_PROXY_PORT)
console.print(f"Proxying agent to port {{DEFAULT_PROXY_PORT}}", style="green")

task_model = SolveTaskModel(
    task=task.to_schema(), device=device.to_schema(), max_steps=args.max_steps
)
agent.solve_task(task_model, follow_logs=True)
"""


def generate_agent(agent_name: str) -> str:
    return f"""
from typing import List, Tuple, Optional
import json
import time
import logging

from rich.console import Console
from rich.json import JSON

from surfkit.llm import LLMProvider
from surfkit.agent import TaskAgent
from taskara import Task

logger: Final = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

console = Console(force_terminal=True)

llm_provider = LLMProvider.from_env()

class {agent_name}(TaskAgent):

    def solve_task(
        self,
        task: Task,
        desktop: Desktop,
        max_steps: int = 10,
        site: Optional[str] = None,
    ) -> Task:
        \"""Solve a task

        Args:
            task (Task): Task to solve
            desktop (Desktop): An AgentDesk desktop instance.
            max_steps (int, optional): Max steps to try and solve. Defaults to 5.
            site (str, optional): Site to open. Defaults to None.

        Returns:
            Task: The task
        \"""

        # > ENTER YOUR TASK LOGIC HERE <

"""


def generate_ci() -> str:
    return ""


def generate_requirements() -> str:
    return """
agentdesk
rich
google-cloud-compute
google-cloud-container
google-cloud-storage
boto3
boto3-stubs
mypy-boto3-ec2
fastapi[all]
pydantic
uvicorn
tenacity
surfkit
"""


# def generate_agentfile(name: str, description: str) -> str:

#     return f"""
# version: v1
# name: "{name}"
# description: "{description}"
# supported_runtimes:
#   - "gke"
# llm_providers:
#   preference:
#     - "gpt-4-vision-preview"
#     - "anthropic/claude-3-opus-20240229"
# public: True
# icon: https://storage.googleapis.com/guisurfer-assets/surf_dino2.webp
# min_cpu: 1
# min_mem: 1Gi
# min_gpu: 0
# """
