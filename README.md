<!-- PROJECT LOGO -->
<br />
<p align="center">
  <!-- <a href="https://github.com/agentsea/skillpacks">
    <img src="https://project-logo.png" alt="Logo" width="80">
  </a> -->

  <h1 align="center">Surfkit</h1>

  <p align="center">
    A toolkit to build GUI surfer AI agents
    <br />
    <a href="https://github.com/agentsea/surfkit"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/agentsea/surfkit">View Demo</a>
    ·
    <a href="https://github.com/agentsea/surfkit/issues">Report Bug</a>
    ·
    <a href="https://github.com/agentsea/surfkit/issues">Request Feature</a>
  </p>
  <br>
</p>

## Installation

```sh
pip install surfkit
```

## Usage

Template a repo with a new agent

```sh
surfkit new [NAME]
```

Run the agent locally

```sh
surfkit create agent --name foo
```

Run the agent on Kubernetes

```sh
surfkit create agent --provider kube
```

List running agents

```sh
surfkit list agents
```

Build a docker container for the agent

```sh
surfkit build
```

Get details about the agent

```sh
surfkit get agent --name foo
```

Get logs for the agent

```sh
surfkit logs --name foo
```

Delete an agent

```sh
surfkit delete agent --name foo
```

Create a device

```sh
surfkit create device --type desktop --provicer gce --name bar
```

List devices

```sh
surfkit list devices
```

View a device

```sh
surfkit view --name bar
```

Delete a device

```sh
surfkit delete device --name bar
```

Solve a task

```sh
surfkit solve --description "search for common french ducks" --agent foo --device bar
```

Solve a task creating the agent adhoc

```sh
surfkit solve --description "search for alpaca sweaters" --device bar --agent-file ./agent.yaml
```

Solve a task killing the agent when the command ends

```sh
surfkit solve --description "search for the meaning of life" --device bar --agent-file ./agent.yaml --kill
```

Login to the hub

```sh
surfkit login
```

Publish the agent

```sh
surfkit publish
```

List published agent types

```sh
surfkit list types
```

Run a published agent

```sh
surfkit create agent --type SurfPizza --runtime kube
```

List tasks

```sh
surfkit list tasks
```

## Developing

Add the following function to your `~/.zshrc` (or similar)

```sh
function sk() {
    local project_dir="/path/to/surfkit/repo"
    local venv_dir="$project_dir/.venv"

    # Save relevant environment variables
    local ssh_auth_sock="$SSH_AUTH_SOCK"
    local ssh_agent_pid="$SSH_AGENT_PID"

    source "$venv_dir/bin/activate"

    # Restore the environment variables
    export SSH_AUTH_SOCK="$ssh_auth_sock"
    export SSH_AGENT_PID="$ssh_agent_pid"

    python -m surfkit.cli.main "$@"
    deactivate
}
```

Replacing `/path/to/surfkit/repo` with the absolute path to your local repo.

Then calling `sk` will execute the working code in your repo from any location.
