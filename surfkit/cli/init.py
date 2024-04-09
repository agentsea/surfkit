import os
import subprocess

from surfkit.cli.templates.agent import (
    generate_agent,
    generate_ci,
    generate_dockerfile,
    generate_main,
    generate_requirements,
    generate_server,
)


def create_git_repository(repo_path):
    """
    Create and initialize a local Git repository.

    Parameters:
    - repo_path: Path where the Git repository will be created.
    """
    os.makedirs(repo_path, exist_ok=True)
    os.chdir(repo_path)
    subprocess.run(["git", "init"], check=True)


def init_agent(name: str):
    create_git_repository(f"./{name}")
    # generate_agentfile(name)
