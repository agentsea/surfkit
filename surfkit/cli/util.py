import subprocess


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
