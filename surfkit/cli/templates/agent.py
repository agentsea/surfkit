import os
from pathlib import Path

from surfkit.cli.util import pkg_from_name


def generate_dockerfile(agent_name: str) -> None:
    out = f"""
FROM thehale/python-poetry:1.8.2-py3.10-slim

COPY . /app
WORKDIR /app

RUN apt-get update && apt-get install -y openssh-client ntp
RUN poetry install

EXPOSE 9090

# Run the application
CMD ["poetry", "run", "python", "-m", "{pkg_from_name(agent_name)}.server"]
"""
    with open(f"Dockerfile", "w") as f:
        f.write(out)


def generate_server(agent_name: str) -> None:
    out = f"""
import logging
import os
import sys
from contextlib import asynccontextmanager
from typing import Final

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from surfkit.server.routes import task_router

from .agent import Agent, router

# Configure logging
logger: Final = logging.getLogger("{pkg_from_name(agent_name)}")
logger.setLevel(int(os.getenv("LOG_LEVEL", str(logging.DEBUG))))
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)

# Ensure logs are flushed immediately
handler.flush = sys.stdout.flush
logger.addHandler(handler)
logger.propagate = False

ALLOW_ORIGINS = os.getenv("ALLOW_ORIGINS", "*").split(",")
ALLOW_METHODS = os.getenv("ALLOW_METHODS", "*").split(",")
ALLOW_HEADERS = os.getenv("ALLOW_HEADERS", "*").split(",")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize the agent type before the server comes live
    Agent.init()
    yield


app = FastAPI(lifespan=lifespan)  # type: ignore

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=ALLOW_METHODS,
    allow_headers=ALLOW_HEADERS,
)

app.include_router(task_router(Agent, router))

if __name__ == "__main__":
    port = os.getenv("SERVER_PORT", "9090")
    reload = os.getenv("SERVER_RELOAD", "false") == "true"
    host = os.getenv("SERVER_HOST", "0.0.0.0")

    uvicorn.run(
        "{pkg_from_name(agent_name)}.server:app",
        host=host,
        port=int(port),
        reload=reload,
        reload_excludes=[".data"],
        log_config=None,
    )
"""
    with open(f"{pkg_from_name(agent_name)}/server.py", "w") as f:
        f.write(out)

    print(f"wrote {pkg_from_name(agent_name)}/server.py")


def generate_agent(agent_name: str, template: str = "surf4v") -> None:
    from .agents.surf4v import Surf4v
    from .agents.surfskelly import SurfSkelly

    if template == "surf4v":
        fourv = Surf4v()
        out = fourv.template(agent_name)
    elif template == "surfskelly":
        skelly = SurfSkelly()
        out = skelly.template(agent_name)
    else:
        raise ValueError(f"Unknown template: {template}")

    with open(f"{pkg_from_name(agent_name)}/agent.py", "w") as f:
        f.write(out)

    print(f"wrote {pkg_from_name(agent_name)}/agent.py")


def generate_dir(agent_name: str) -> None:
    os.mkdir(pkg_from_name(agent_name))


def generate_pyproject(agent_name: str, description, git_user_ref: str) -> None:
    out = f"""
[tool.poetry]
name = "{agent_name}"
version = "0.1.0"
description = "AI agent for {description}"
authors = ["{git_user_ref}"]
license = "MIT"
readme = "README.md"
packages = [{{include = "{pkg_from_name(agent_name)}"}}]

[tool.poetry.dependencies]
python = "^3.10"
sqlalchemy = "^2.0.27"
pydantic = "^2.6.3"
requests = "^2.31.0"
fastapi = {{version = "^0.109", extras = ["all"]}}
surfkit = "^0.1.195"


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

    print("wrote pyproject.toml")


def generate_agentfile(
    name: str, description: str, image_repo: str, icon_url: str
) -> None:

    out = f"""
api_version: v1
kind: TaskAgent
name: "{name}"
description: "{description}"
cmd: "poetry run python -m {pkg_from_name(name)}.server"
img_repo: "{image_repo}"
tags:
  - "gui"
supports:
  - "desktop"
versions:
  latest: "{image_repo}:latest"
runtimes:
  - type: "agent"
    preference:
      - "process"
      - "docker"
      - "kube"
llm_providers:
  preference:
    - "gpt-4o"
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

    print("wrote agent.yaml")


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
.agentsea
"""
    file_path = Path(".gitignore")

    if file_path.exists():
        with file_path.open("a") as file:
            file.write("\ndata/\n")
    else:
        with file_path.open("w") as file:
            file.write(out)

    print("wrote .gitignore")


def generate_readme(agent_name: str, description: str) -> None:

    out = f"""# {agent_name}

{description}

## Install
```sh
pip install surfkit
```

## Usage

Create an agent
```sh
surfkit create agent -f ./agent.yaml --runtime {{ process | docker | kube }} --name foo
```

List running agents
```sh
surfkit list agents
```

Use the agent to solve a task
```sh
surfkit solve --agent foo --description "Search for french ducks" --device-type desktop
```

Get the agent logs
```sh
surfkit logs --name foo
```

Delete the agent
```sh
surfkit delete agent --name foo
```

"""
    file_path = Path("README.md")

    if not file_path.exists():
        with file_path.open("w") as file:
            file.write(out)

    print("wrote README.md")
