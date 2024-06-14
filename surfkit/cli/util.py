import subprocess
import sys
from typing import Optional

from taskara.runtime.base import Tracker, TrackerRuntime

from surfkit.util import find_open_port


def get_git_global_user_config():
    # Command to get the global user name
    name_command = ["git", "config", "--global", "user.name"]
    # Command to get the global user email
    email_command = ["git", "config", "--global", "user.email"]

    try:
        # Execute the commands
        name = subprocess.run(
            name_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        ).stdout.strip()
        email = subprocess.run(
            email_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        ).stdout.strip()

        return f"{name} <{email}>"
    except subprocess.CalledProcessError as e:
        print("Error getting git global user config: ", e)
        raise


def pkg_from_name(name: str) -> str:
    """Return package name from module name"""
    name = name.replace("-", "_")
    return name.lower()


def is_poetry_installed():
    """Check if 'poetry' command is available on the system."""
    try:
        subprocess.run(
            ["poetry", "--version"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return True
    except subprocess.CalledProcessError:
        return False
    except FileNotFoundError:
        return False


def run_poetry_install():
    """Run 'poetry install' using the subprocess module."""
    try:
        print("Running 'poetry install'...")
        subprocess.run(["poetry", "install"], check=True)
        print("'poetry install' executed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running 'poetry install': {e}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise


def is_docker_installed():
    """Check if 'docker' command is available on the system."""
    try:
        subprocess.run(
            ["docker", "version"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return True
    except subprocess.CalledProcessError:
        return False
    except FileNotFoundError:
        return False


def build_docker_image(
    dockerfile_path: str,
    tag: str,
    push: bool = True,
    builder: str = "surfbuilder",
    platforms: str = "linux/amd64,linux/arm64",
):
    try:
        # Check Docker version
        result = subprocess.run(
            ["docker", "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if result.returncode != 0:
            raise RuntimeError("Docker is not installed or not running.")
        print(result.stdout.decode().strip())

        # Ensure using the correct Docker context
        result = subprocess.run(
            ["docker", "context", "use", "default"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Error setting Docker context: {result.stderr.decode()}"
            )

        # Check if the buildx builder exists
        result = subprocess.run(
            ["docker", "buildx", "inspect", builder],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if result.returncode != 0:
            print(f"Builder '{builder}' not found. Creating a new builder.")
            result = subprocess.run(
                ["docker", "buildx", "create", "--name", builder, "--use"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"Error creating buildx builder: {result.stderr.decode()}"
                )
        else:
            # Use the existing builder
            result = subprocess.run(
                ["docker", "buildx", "use", builder],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"Error using buildx builder: {result.stderr.decode()}"
                )

        # Ensure the builder is bootstrapped
        result = subprocess.run(
            ["docker", "buildx", "inspect", "--bootstrap"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Error bootstrapping buildx builder: {result.stderr.decode()}"
            )

        # Prepare the command for building the image
        command = ["docker", "buildx", "build"]
        if push:
            command.append("--push")
        command.extend(
            ["--platform", platforms, "--tag", tag, "--file", dockerfile_path, "."]
        )

        # Building (and optionally pushing) the Docker image
        result = subprocess.run(
            command, check=True, stdout=sys.stdout, stderr=subprocess.STDOUT
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Error building the Docker image: {result.stderr.decode()}"
            )

        print(
            f"Docker image tagged as {tag} has been successfully built{' and pushed' if push else ''} for platforms {platforms}."
        )

    except subprocess.CalledProcessError as e:
        print(f"Subprocess error: {e.stderr.decode() if e.stderr else str(e)}")
        raise
    except RuntimeError as e:
        print(f"Runtime error: {e}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise


def tracker_addr_agent(
    tracker: Tracker,
    agent_runtime: str,
) -> str:
    if agent_runtime == "process":
        if tracker.runtime.name() == "process":
            return tracker.runtime.runtime_local_addr(tracker.name, tracker.owner_id)
        elif tracker.runtime.name() == "docker":
            return f"http://localhost:{tracker.port}"
        elif tracker.runtime.name() == "kube":
            port = find_open_port(9070, 9090)
            if not port:
                raise Exception("No open port found for tracker")
            tracker.proxy(port)
            return f"http://localhost:{port}"
        else:
            raise ValueError(f"Unknown agent runtime: {agent_runtime}")
    elif agent_runtime == "docker":
        if tracker.runtime.name() == "process":
            raise ValueError("Cannot use Docker agent with a process tracker")
        elif tracker.runtime.name() == "docker":
            return tracker.runtime.runtime_local_addr(tracker.name, tracker.owner_id)
        elif tracker.runtime.name() == "kube":
            raise ValueError("Cannot use Docker agent with a Kubernetes tracker")
        else:
            raise ValueError(f"Unknown agent runtime: {agent_runtime}")
    elif agent_runtime == "kube":
        if tracker.runtime.name() == "process":
            raise ValueError("Cannot use Kubernetes agent with a process tracker")
        elif tracker.runtime.name() == "docker":
            raise ValueError("Cannot use Kubernetes agent with a Docker tracker")
        elif tracker.runtime.name() == "kube":
            return tracker.runtime.runtime_local_addr(tracker.name, tracker.owner_id)
        else:
            raise ValueError(f"Unknown agent runtime: {agent_runtime}")
    else:
        raise ValueError(f"Unknown agent runtime: {agent_runtime}")


def tracker_addr_local(
    tracker: Tracker,
) -> str:
    local_port = tracker.port
    if tracker.runtime.requires_proxy():
        local_port = find_open_port(9070, 10070)
        if not local_port:
            raise SystemError("No available ports found")
        tracker.proxy(local_port=local_port)
    return f"http://localhost:{local_port}"
