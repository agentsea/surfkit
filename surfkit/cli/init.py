import os
import subprocess

from surfkit.cli.templates.agent import (
    generate_agentfile,
    generate_agent,
    generate_ci,
    generate_dockerfile,
    generate_main,
    generate_requirements,
    generate_server,
)


def create_git_repository(repo_path, add_files=False, commit_message="Initial commit"):
    """
    Create and initialize a local Git repository.

    Parameters:
    - repo_path: Path where the Git repository will be created.
    - add_files: Boolean indicating whether to add files in the repository to the staging area.
    - commit_message: Commit message to use if committing files.
    """
    # Ensure the directory exists (create it if not)
    os.makedirs(repo_path, exist_ok=True)

    # Change to the directory
    os.chdir(repo_path)

    # Initialize the Git repository
    subprocess.run(["git", "init"], check=True)


def init_agent(name: str):
    pass
