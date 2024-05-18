<!-- PROJECT LOGO -->
<br />
<p align="center">
  <!-- <a href="https://github.com/agentsea/skillpacks">
    <img src="https://project-logo.png" alt="Logo" width="80">
  </a> -->

  <h1 align="center">Surfkit</h1>

  <p align="center">
    A toolkit for building AI agents that use devices
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

### Building Agents

Initialize a new project

```sh
surfkit new [NAME]
```

Build a docker container for the agent

```sh
surfkit build
```

### Running Agents

Create an agent locally

```sh
surfkit create agent --name foo
```

Create an agent on kubernetes

```sh
surfkit create agent --runtime kube
```

List running agents

```sh
surfkit list agents
```

Get details about a specific agent

```sh
surfkit get agent --name foo
```

Fetch logs for a specific agent

```sh
surfkit logs --name foo
```

Delete an agent

```sh
surfkit delete agent --name foo
```

### Managing Devices

Create a device

```sh
surfkit create device --type desktop --provicer gce --name bar
```

List devices

```sh
surfkit list devices
```

View device in UI

```sh
surfkit view --name bar
```

Delete a device

```sh
surfkit delete device --name bar
```

### Tracking Tasks

Create a tracker

```sh
surfkit create tracker
```

List trackers

```sh
surfkit list trackers
```

Delete a tracker

```sh
surfkit delete tracker -n foo
```

### Solving Tasks

Solve a task with an existing setup

```sh
surfkit solve --description "search for common french ducks" --agent foo --device bar
```

Solve a task creating the agent ad hoc

```sh
surfkit solve --description "search for alpaca sweaters" \
--device bar --agent-file ./agent.yaml
```

Solve a task and kill the agent post-execution

```sh
surfkit solve --description "search for the meaning of life" \
--device bar --agent-file ./agent.yaml --kill
```

List tasks

```sh
surfkit list tasks
```

### Publishing Agents

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

## Developing

Add the following function to your `~/.zshrc` (or similar)

```sh
function sk() {
    local project_dir="/path/to/surfkit/repo"
    local venv_dir="$project_dir/.venv"

    local ssh_auth_sock="$SSH_AUTH_SOCK"
    local ssh_agent_pid="$SSH_AGENT_PID"

    source "$venv_dir/bin/activate"

    export SSH_AUTH_SOCK="$ssh_auth_sock"
    export SSH_AGENT_PID="$ssh_agent_pid"

    python -m surfkit.cli.main "$@"
    deactivate
}
```

Replacing `/path/to/surfkit/repo` with the absolute path to your local repo.

Then calling `sk` will execute the working code in your repo from any location.
