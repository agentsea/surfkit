import os
import subprocess

from surfkit.cli.templates.agent import (
    generate_agent,
    generate_dockerfile,
    generate_main,
    generate_pyproject,
    generate_server,
    generate_agentfile,
    generate_dir,
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


def new_agent(
    name: str, description: str, git_user_ref: str, img_repo: str, icon_url: str
) -> None:
    generate_dir(name)
    generate_dockerfile(name)
    generate_pyproject(name, description, git_user_ref)
    generate_agent(name)
    generate_server(name)
    generate_main(name)
    generate_agentfile(
        name, description=description, image_repo=img_repo, icon_url=icon_url
    )
