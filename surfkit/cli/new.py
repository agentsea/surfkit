import os
import subprocess

from surfkit.cli.templates.agent import (
    generate_agent,
    generate_agentfile,
    generate_dir,
    generate_dockerfile,
    generate_gitignore,
    generate_pyproject,
    generate_readme,
    generate_server,
)

from .util import (
    is_docker_installed,
    is_poetry_installed,
    pkg_from_name,
    run_poetry_install,
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
    name: str,
    description: str,
    git_user_ref: str,
    img_repo: str,
    icon_url: str,
    template: str,
) -> None:
    if not is_poetry_installed():
        raise SystemError(
            "Poetry not found on system, please install at https://python-poetry.org/docs/#installation"
        )

    if not is_docker_installed():
        raise SystemError(
            "Docker not found on system, please install at https://docs.docker.com/engine/install/"
        )

    generate_dir(name)
    generate_dockerfile(name)
    generate_pyproject(name, description, git_user_ref)
    generate_agent(name, template)
    generate_server(name)
    generate_gitignore()
    generate_agentfile(
        name, description=description, image_repo=img_repo, icon_url=icon_url
    )
    generate_readme(name, description)

    run_poetry_install()
