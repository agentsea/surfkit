import os
from pathlib import Path


def generate_dockerfile(agent_name: str) -> None:
    out = f"""
FROM thehale/python-poetry:1.8.2-py3.10-slim

COPY . /app
WORKDIR /app

RUN apt-get update && apt-get install -y openssh-client ntp
RUN poetry install

EXPOSE 9090

# Run the application
CMD ["uvicorn", "{agent_name.lower()}.server:app", "--host=0.0.0.0", "--port=9090", "--log-level", "debug"]
"""
    with open(f"Dockerfile", "w") as f:
        f.write(out)


def generate_server(agent_name: str) -> None:
    out = f"""
import os

from taskara.server.models import SolveTaskModel, TaskModel, TasksModel
from taskara.task import Task
from surfkit.hub import Hub
from surfkit.llm import LLMProvider
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from tenacity import retry, stop_after_attempt, wait_fixed

from .agent import Agent


# Get the LLM provider from env
llm_provider = LLMProvider.from_env()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize the agent type before the server comes live
    Agent.init()
    yield


app = FastAPI(lifespan=lifespan)  # type: ignore

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {{"message": "Agent in the shell"}}


@app.get("/health")
async def health():
    return {{"status": "ok"}}


@app.post("/v1/tasks")
async def solve_task(background_tasks: BackgroundTasks, task_model: SolveTaskModel):
    print(f"solving task: \n{{task_model.model_dump()}}")
    try:
        # TODO: we need to find a way to do this earlier but get status back
        llm_provider.check_model()
    except Exception as e:
        print(f"Cannot connect to LLM providers: {{e}} -- did you provide a valid key?")
        return {{
            "status": "failed",
            "message": f"failed to conect to LLM providers: {{e}} -- did you provide a valid key?",
        }}

    background_tasks.add_task(_solve_task, task_model)
    print("created background task...")


def _solve_task(task_model: SolveTaskModel):
    task = Task.from_schema(task_model.task, owner_id="local")
    if task.remote:
        print("connecting to remote task...")
        HUB_SERVER = os.environ.get("SURF_HUB_SERVER", "https://surf.agentlabs.xyz")
        HUB_API_KEY = os.environ.get("HUB_API_KEY")
        if not HUB_API_KEY:
            raise Exception("$HUB_API_KEY not set")

        hub = Hub(HUB_SERVER)
        user_info = hub.get_user_info(HUB_API_KEY)
        print("got user info: ", user_info.__dict__)

        task = get_remote_task(
            id=task.id,
            owner_id=user_info.email,  # type: ignore
            server=task.remote,
        )
        print("got remote task: ", task.__dict__)

    print("Saving remote tasks status to running...")
    task.status = "in progress"
    task.save()

    if task_model.device:
        print(f"connecting to device {{task_model.device.name}}...")
        device = None
        for Device in Agent.supported_devices():
            if Device.name() == task_model.device.name:
                print("found device: ", task_model.device.model_dump())
                print("model config: ", task_model.device.config)
                config = Device.connect_config_type()(**task_model.device.config)
                device = Device.connect(config=config)

        if not device:
            raise ValueError(
                f"Device {{task_model.device.name}} provided in solve task, but not supported by agent"
            )

        print("connected to device: ", device.__dict__)
    else:
        raise ValueError("No device provided")

    print("starting agent...")
    if task_model.agent:
        config = Agent.config_type()(**task_model.agent.config.model_dump())
        agent = Agent.from_config(config=config)
    else:
        agent = Agent.default()

    try:
        fin_task = agent.solve_task(task=task, device=device, max_steps=task.max_steps)
    except Exception as e:
        print("error running agent: ", e)
        task.status = "failed"
        task.error = str(e)
        task.save()
        task.post_message("assistant", f"Failed to run task '{{task.description}}': {{e}}")
        raise e
    if fin_task:
        fin_task.save()


@app.get("/v1/tasks", response_model=TasksModel)
async def get_tasks():
    tasks = Task.find()
    return TasksModel(tasks=[task.to_schema() for task in tasks])


@app.get("/v1/tasks/{{id}}", response_model=TaskModel)
async def get_task(id: str):
    tasks = Task.find(id=id)
    if not tasks:
        raise Exception(f"Task {{id}} not found")
    return tasks[0].to_schema()

    
@retry(stop=stop_after_attempt(10), wait=wait_fixed(10))
def get_remote_task(id: str, owner_id: str, server: str) -> Task:
    HUB_API_KEY = os.environ.get("HUB_API_KEY")
    if not HUB_API_KEY:
        raise Exception("$HUB_API_KEY not set")

    print("connecting to remote task: ", id, HUB_API_KEY)
    try:
        tasks = Task.find(
            id=id,
            remote=server,
            owner_id=owner_id,
        )
        if not tasks:
            raise Exception(f"Task {{id}} not found")
        print("got remote task: ", tasks[0].__dict__)
        return tasks[0]
    except Exception as e:
        print("error getting remote task: ", e)
        raise e

        
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9090, reload=True)
"""
    with open(f"{agent_name.lower()}/server.py", "w") as f:
        f.write(out)


def generate_main(agent_name: str) -> None:
    out = f"""
import argparse
import logging
import yaml

from rich.console import Console
from agentdesk.device import Desktop, ProvisionConfig
from agentdesk.vm.gce import GCEProvider
from agentdesk.vm.ec2 import EC2Provider
from agentdesk.vm.qemu import QemuProvider
from taskara import Task
from taskara.server.models import SolveTaskModel
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

# TODO: dynamically supply and provision devices
console.print("Creating device...", style="green")
print("provider data: ", provider.to_data())
config = ProvisionConfig(provider=provider.to_data())
device: Desktop = Desktop.ensure("gpt4v-demo", config=config)

# View the desktop, we'll run in the background so it doesn't block
device.view(background=True)

if args.agent_runtime == "docker":
    print("using docker runtime...")
    dconf = DockerConnectConfig()
    conf = AgentRuntimeConfig(provider="docker", docker_config=dconf)
elif args.agent_runtime == "kube":
    print("using kube runtime...")
    kconf = KubeConnectConfig(namespace=args.namespace)
    conf = AgentRuntimeConfig(provider="kube", kube_config=kconf)
else:
    raise ValueError("Unknown agent runtime")

runtime = load_agent_runtime(conf)

print("device schema: ", device.to_schema())
console.print("Running agent...", style="green")
agent = runtime.run(agent_type, args.name, args.version, llm_providers_local=True)

agent.proxy(DEFAULT_PROXY_PORT)
console.print(f"Proxying agent to port {{DEFAULT_PROXY_PORT}}", style="green")

task_model = SolveTaskModel(
    task=task.to_schema(), device=device.to_schema(), max_steps=args.max_steps
)
agent.solve_task(task_model, follow_logs=True)

"""
    with open(f"{agent_name.lower()}/__main__.py", "w") as f:
        f.write(out)


def generate_agent(agent_name: str) -> None:
    out = f"""
from typing import List, Type
import logging
from typing import Final

from devicebay import Device
from agentdesk.device import Desktop
from rich.console import Console
from surfkit.llm import LLMProvider
from pydantic import BaseModel
from surfkit.agent import TaskAgent
from taskara import Task

logger: Final = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

console = Console(force_terminal=True)

llm_provider = LLMProvider.from_env()


class {agent_name}Config(BaseModel):
    pass


class {agent_name}(TaskAgent):
    \"""A desktop agent that uses GPT-4V augmented with OCR and Grounding Dino to solve tasks\"""

    def solve_task(
        self,
        task: Task,
        device: Device,
        max_steps: int = 30,
    ) -> Task:
        \"""Solve a task

        Args:
            task (Task): Task to solve.
            device (Desktop): Device to perform the task on.
            max_steps (int, optional): Max steps to try and solve. Defaults to 30.

        Returns:
            Task: The task
        \"""

        task.post_message("assistant", f"Starting task '{{task.description}}'")
        # > ENTER YOUR TASK LOGIC HERE <

        
    @classmethod
    def supported_devices(cls) -> List[Type[Device]]:
        \"""Devices this agent supports

        Returns:
            List[Type[Device]]: A list of supported devices
        \"""
        return [Desktop]

    @classmethod
    def config_type(cls) -> Type[{agent_name}Config]:
        \"""Type of config

        Returns:
            Type[DinoConfig]: Config type
        \"""
        return {agent_name}Config

    @classmethod
    def from_config(cls, config: {agent_name}Config) -> "{agent_name}":
        \"""Create an agent from a config

        Args:
            config (DinoConfig): Agent config

        Returns:
            {agent_name}: The agent
        \"""
        return {agent_name}()

    @classmethod
    def default(cls) -> "{agent_name}":
        \"""Create a default agent

        Returns:
            {agent_name}: The agent
        \"""
        return {agent_name}()

    @classmethod
    def init(cls) -> None:
        \"""Initialize the agent class\"""
        # <INITIALIZE AGENT HERE>
        return


Agent = {agent_name}
"""
    with open(f"{agent_name.lower()}/agent.py", "w") as f:
        f.write(out)


def generate_dir(agent_name: str) -> None:
    os.mkdir(agent_name.lower())


def generate_pyproject(agent_name: str, description, git_user_ref: str) -> None:
    out = f"""
[tool.poetry]
name = "{agent_name}"
version = "0.1.0"
description = "AI agent for {description}"
authors = ["{git_user_ref}"]
license = "MIT"
readme = "README.md"
packages = [{{include = "{agent_name}"}}]

[tool.poetry.dependencies]
python = "^3.10"
sqlalchemy = "^2.0.27"
pydantic = "^2.6.3"
requests = "^2.31.0"
surfkit = "^0.1.92"
tenacity = "^8.2.3"
fastapi = {{version = "^0.109", extras = ["all"]}}


[tool.poetry.group.dev.dependencies]
flake8 = "^7.0.0"
black = "^24.2.0"
pytest = "^8.0.2"
ipykernel = "^6.29.3"
pytest-env = "^1.1.3"

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"
"""
    with open(f"pyproject.toml", "w") as f:
        f.write(out)


def generate_agentfile(
    name: str, description: str, image_repo: str, icon_url: str
) -> None:

    out = f"""
api_version: v1alpha
kind: TaskAgent
name: "{name}"
description: "{description}"
cmd: "poetry run python -m {name.lower()}.server"
image: "{image_repo}:latest"
versions:
  latest: "{image_repo}:latest"
runtimes:
  - type: "agent"
    preference:
      - "venv"
      - "docker"
      - "kube"
llm_providers:
  preference:
    - "gpt-4-turbo"
    - "anthropic/claude-3-opus-20240229"
public: True
icon: {icon_url}
resource_requests:
  cpu: "1"
  memory: "2Gi"
resource_limits:
  cpu: "2"
  memory: "4Gi"
"""
    with open(f"agent.yaml", "w") as f:
        f.write(out)


def generate_gitignore() -> None:

    out = """
# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# C extensions
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
#  Usually these files are written by a python script from a template
#  before PyInstaller builds the exe, so as to inject date/other infos into it.
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.py,cover
.hypothesis/
.pytest_cache/
cover/

# Translations
*.mo
*.pot

# Django stuff:
*.log
local_settings.py
db.sqlite3
db.sqlite3-journal

# Flask stuff:
instance/
.webassets-cache

# Scrapy stuff:
.scrapy

# Sphinx documentation
docs/_build/

# PyBuilder
.pybuilder/
target/

# Jupyter Notebook
.ipynb_checkpoints

# IPython
profile_default/
ipython_config.py

# pyenv
#   For a library or package, you might want to ignore these files since the code is
#   intended to run in multiple environments; otherwise, check them in:
# .python-version

# pipenv
#   According to pypa/pipenv#598, it is recommended to include Pipfile.lock in version control.
#   However, in case of collaboration, if having platform-specific dependencies or dependencies
#   having no cross-platform support, pipenv may install dependencies that don't work, or not
#   install all needed dependencies.
#Pipfile.lock

# poetry
#   Similar to Pipfile.lock, it is generally recommended to include poetry.lock in version control.
#   This is especially recommended for binary packages to ensure reproducibility, and is more
#   commonly ignored for libraries.
#   https://python-poetry.org/docs/basic-usage/#commit-your-poetrylock-file-to-version-control
#poetry.lock

# pdm
#   Similar to Pipfile.lock, it is generally recommended to include pdm.lock in version control.
#pdm.lock
#   pdm stores project-wide configurations in .pdm.toml, but it is recommended to not include it
#   in version control.
#   https://pdm.fming.dev/#use-with-ide
.pdm.toml

# PEP 582; used by e.g. github.com/David-OConnor/pyflow and github.com/pdm-project/pdm
__pypackages__/

# Celery stuff
celerybeat-schedule
celerybeat.pid

# SageMath parsed files
*.sage.py

# Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Spyder project settings
.spyderproject
.spyproject

# Rope project settings
.ropeproject

# mkdocs documentation
/site

# mypy
.mypy_cache/
.dmypy.json
dmypy.json

# Pyre type checker
.pyre/

# pytype static type analyzer
.pytype/

# Cython debug symbols
cython_debug/

# PyCharm
#  JetBrains specific template is maintained in a separate JetBrains.gitignore that can
#  be found at https://github.com/github/gitignore/blob/main/Global/JetBrains.gitignore
#  and can be added to the global gitignore or merged into this file.  For a more nuclear
#  option (not recommended) you can uncomment the following to ignore the entire idea folder.
#.idea/

.data/
cidata.iso
"""
    file_path = Path(".gitignore")

    if file_path.exists():
        with file_path.open("a") as file:
            file.write("\ndata/\n")
    else:
        with file_path.open("w") as file:
            file.write(out)
