import subprocess
import sys


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
        # Ensure using the correct Docker context
        subprocess.run(
            ["docker", "context", "use", "default"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Create or use an existing buildx builder that supports multi-arch
        result = subprocess.run(
            ["docker", "buildx", "create", "--name", builder, "--use", "--append"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if result.returncode != 0:
            # If creation failed because it already exists, just use it
            subprocess.run(
                ["docker", "buildx", "use", builder],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
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
        print(
            f"Docker image tagged as {tag} has been successfully built{' and pushed' if push else ''} for platforms {platforms}."
        )

    except subprocess.CalledProcessError as e:
        print(
            f"An error occurred while building {'and pushing ' if push else ''}the Docker image for platforms {platforms}: {e.stderr.decode() if e.stderr else None}"
        )
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
