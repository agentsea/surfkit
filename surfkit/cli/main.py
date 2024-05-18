import webbrowser
from urllib.parse import urljoin
from typing import Optional
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as pkgversion
from pdb import run
from typing import Optional
from urllib.parse import urljoin

import requests
import rich
import typer
import yaml
from namesgenerator import get_random_name
from tabulate import tabulate

from surfkit.runtime.agent.base import AgentInstance

art = """
 _______                ___  __  __  __  __   
|     __|.--.--..----..'  _||  |/  ||__||  |_ 
|__     ||  |  ||   _||   _||     < |  ||   _|
|_______||_____||__|  |__|  |__|\__||__||____|
"""

app = typer.Typer()

# Sub-command groups
create_group = typer.Typer(help="Create resources")
list_group = typer.Typer(help="List resources")
get_group = typer.Typer(help="Get resources")
view_group = typer.Typer(help="View resources")
delete_group = typer.Typer(help="Delete resources")
clean_group = typer.Typer(help="Clean resources")

app.add_typer(create_group, name="create")
app.add_typer(list_group, name="list")
app.add_typer(get_group, name="get")
app.add_typer(view_group, name="view")
app.add_typer(delete_group, name="delete")
# app.add_typer(clean_group, name="clean")


# Callback for showing help
def show_help(ctx: typer.Context, command_group: str):
    if ctx.invoked_subcommand is None:
        if command_group == "root":
            pass
            # typer.echo(art)
        typer.echo(ctx.get_help())
        raise typer.Exit()


try:
    __version__ = pkgversion("surfkit")
except PackageNotFoundError:
    # Fallback version or error handling
    __version__ = "unknown"


@app.command(help="Show the version of the CLI")
def version():
    """Show the CLI version."""
    typer.echo(__version__)


# Root command callback
@app.callback(invoke_without_command=True)
def default(ctx: typer.Context):
    show_help(ctx, "root")


# 'create' command group callback
@create_group.callback(invoke_without_command=True)
def create_default(ctx: typer.Context):
    show_help(ctx, "create")


# 'list' command group callback
@list_group.callback(invoke_without_command=True)
def list_default(ctx: typer.Context):
    show_help(ctx, "list")


# 'get' command group callback
@get_group.callback(invoke_without_command=True)
def get_default(ctx: typer.Context):
    show_help(ctx, "get")


# 'delete' command group callback
@delete_group.callback(invoke_without_command=True)
def delete_default(ctx: typer.Context):
    show_help(ctx, "delete")


# 'view' command group callback
@view_group.callback(invoke_without_command=True)
def view_default(ctx: typer.Context):
    show_help(ctx, "view")


# 'clean' command group callback
# @clean_group.callback(invoke_without_command=True)
# def clean_default(ctx: typer.Context):
#     show_help(ctx, "clean")


# 'create' sub-commands
@create_group.command("device")
def create_device(
    name: Optional[str] = typer.Option(
        None,
        "--name",
        "-n",
        help="The name of the desktop to create. Defaults to a generated name.",
    ),
    type: Optional[str] = typer.Option(
        "desktop",
        "--type",
        "-t",
        help="The type of device to create. Options are 'desktop'",
    ),
    provider: str = typer.Option(
        "qemu",
        "--provider",
        "-p",
        help="The provider type for the desktop. Options are 'ec2', 'gce', and 'qemu'",
    ),
    image: Optional[str] = typer.Option(
        None, help="The image to use for the desktop. Defaults to Ubuntu Jammy."
    ),
    memory: int = typer.Option(4, help="The amount of memory (in GB) for the desktop."),
    cpu: int = typer.Option(2, help="The number of CPU cores for the desktop."),
    disk: str = typer.Option(
        "30gb",
        help="The disk size for the desktop. Format as '<size>gb'.",
    ),
    reserve_ip: bool = typer.Option(
        False,
        help="Whether to reserve an IP address for the desktop.",
    ),
):
    from agentdesk.server.models import V1ProviderData
    from agentdesk.vm.load import load_provider

    if type != "desktop":
        typer.echo("Currently only 'desktop' type is supported.")
        raise typer.Exit()

    if not name:
        name = get_random_name(sep="-")

    if provider == "ec2":
        data = V1ProviderData(type=provider, args={"region": "us-east-1"})
        _provider = load_provider(data)

    else:
        data = V1ProviderData(type=provider)
        _provider = load_provider(data)

    typer.echo(f"Creating desktop '{name}' using '{provider}' provider")
    try:
        _provider.create(
            name=name,
            image=image,
            memory=memory,
            cpu=cpu,
            disk=disk,
            reserve_ip=reserve_ip,
        )
    except KeyboardInterrupt:
        print("Keyboard interrupt received, exiting...")
        return
    except:
        raise


@create_group.command("tracker")
def create_tracker(
    name: Optional[str] = typer.Option(
        None,
        "--name",
        "-n",
        help="The name of the tracker to create. Defaults to a generated name.",
    ),
    runtime: str = typer.Option(
        "docker",
        "--runtime",
        "-r",
        help="The runtime to use for the tracker. Options are 'docker' or 'kube'.",
    ),
    image: str = typer.Option(
        "us-central1-docker.pkg.dev/agentsea-dev/taskara/api:latest",
        "--image",
        "-i",
        help="The Docker image to use for the tracker.",
    ),
):

    if runtime == "docker":
        from taskara.runtime.docker import DockerTrackerRuntime, DockerConnectConfig

        runt = DockerTrackerRuntime(DockerConnectConfig(image=image))

    elif runtime == "kube":
        from taskara.runtime.kube import KubeTrackerRuntime, KubeConnectConfig

        runt = KubeTrackerRuntime(KubeConnectConfig(image=image))

    else:
        typer.echo(f"Invalid runtime: {runtime}")
        raise typer.Exit()

    if not name:
        name = get_random_name(sep="-")
        if not name:
            raise SystemError("Name is required for tracker")

    server = runt.run(name=name, auth_enabled=False)
    typer.echo(f"Tracker '{name}' created using '{runtime}' runtime")


@create_group.command("agent")
def create_agent(
    runtime: str = typer.Option(
        None,
        "--runtime",
        "-r",
        help="The runtime to use. Options are 'process', 'docker', or 'kube'",
    ),
    file: str = typer.Option(
        "./agent.yaml", "--file", "-f", help="Path to the agent configuration file."
    ),
    name: Optional[str] = typer.Option(
        None, "--name", "-n", help="Name of the agent. Defaults to a generated name."
    ),
    type: Optional[str] = typer.Option(
        None, "--type", "-t", help="Type of the agent if predefined."
    ),
):

    from surfkit.server.models import V1AgentType
    from surfkit.types import AgentType

    if not runtime:
        if file:
            runtime = "process"
        else:
            runtime = "docker"

    if runtime == "docker":
        from surfkit.runtime.agent.docker import DockerAgentRuntime, DockerConnectConfig

        conf = DockerConnectConfig()
        runt = DockerAgentRuntime.connect(cfg=conf)

    elif runtime == "kube":
        from surfkit.runtime.agent.kube import KubeAgentRuntime, KubeConnectConfig

        conf = KubeConnectConfig()
        runt = KubeAgentRuntime.connect(cfg=conf)

    elif runtime == "process":
        from surfkit.runtime.agent.process import (
            ProcessAgentRuntime,
            ProcessConnectConfig,
        )

        conf = ProcessConnectConfig()
        runt = ProcessAgentRuntime.connect(cfg=conf)

    else:
        raise ValueError(f"Unknown runtime '{runtime}'")

    if type:
        from surfkit.config import HUB_API_URL

        types = AgentType.find(remote=HUB_API_URL, name=type)
        if not types:
            raise ValueError(f"Agent type '{type}' not found")
        agent_type = types[0]
    else:
        try:
            with open(file, "r") as f:
                data = yaml.safe_load(f)
                agent_type_v1 = V1AgentType(**data)
        except Exception as e:
            raise ValueError(f"Failed to parse find or parse agent.yaml: {e}")

        agent_type = AgentType.from_v1(agent_type_v1)

        types = AgentType.find(name=agent_type.name)
        if types:
            typ = types[0]
            typ.update(agent_type_v1)
        else:
            agent_type.save()

    if not name:
        from surfkit.runtime.agent.util import instance_name

        name = instance_name(agent_type)

    if type:
        typer.echo(
            f"Running agent '{type}' with runtime '{runtime}' and name '{name}'..."
        )
    else:
        typer.echo(
            f"Running agent '{file}' with runtime '{runtime}' and name '{name}'..."
        )

    runt.run(agent_type, name)
    typer.echo(f"Successfully created agent '{name}'")


# 'list' sub-commands


@list_group.command("agents")
def list_agents(
    runtime: Optional[str] = typer.Option(
        None,
        "--runtime",
        "-r",
        help="List agent directly from the runtime. Options are 'docker', 'kube', or 'all' (default)",
    ),
):
    agents_list = []

    if runtime:
        if runtime == "docker" or runtime == "all":
            from surfkit.runtime.agent.docker import (
                DockerAgentRuntime,
                DockerConnectConfig,
            )

            try:
                dconf = DockerConnectConfig()
                runt = DockerAgentRuntime.connect(cfg=dconf)
                agents = runt.list()
                for agent in agents:
                    agents_list.append([agent.name, agent.type, "docker", agent.port])
            except Exception as e:
                if runtime != "all":
                    raise
                print(f"Failed to list agents for Docker runtime: {e}")

        if runtime == "kube" or runtime == "all":
            from surfkit.runtime.agent.kube import KubeAgentRuntime, KubeConnectConfig

            try:
                kconf = KubeConnectConfig()
                runt = KubeAgentRuntime.connect(cfg=kconf)
                agents = runt.list()
                for agent in agents:
                    agents_list.append([agent.name, agent.type, "kube", agent.port])
            except Exception as e:
                if runtime != "all":
                    raise
                print(f"Failed to list agents for Kubernetes runtime: {e}")

        if runtime == "process" or runtime == "all":
            from surfkit.runtime.agent.process import (
                ProcessAgentRuntime,
                ProcessConnectConfig,
            )

            try:
                pconf = ProcessConnectConfig()
                runt = ProcessAgentRuntime.connect(cfg=pconf)
                agents = runt.list()
                for agent in agents:
                    agents_list.append(
                        [
                            agent.name,
                            agent.type.kind,
                            agent.type.name,
                            "process",
                            agent.port,
                        ]
                    )
            except Exception as e:
                if runtime != "all":
                    raise
                print(f"Failed to list agents for Process runtime: {e}")

    else:
        from surfkit.runtime.agent.base import AgentInstance

        agents = AgentInstance.find()
        for agent in agents:
            agents_list.append(
                [
                    agent.name,
                    agent.type.kind,
                    agent.type.name,
                    agent.runtime.name(),
                    agent.status,
                    agent.port,
                ]
            )

    # Print the collected data from all or a single runtime
    if agents_list:
        print(
            tabulate(
                agents_list,
                headers=[
                    "Name",
                    "Kind",
                    "Type",
                    "Runtime",
                    "Status",
                    "Port",
                ],
            )
        )
        print("")
    else:
        print("No agents found.")


@list_group.command("devices")
def list_devices(
    provider: Optional[str] = typer.Option(
        None, "--provider", "-p", help="The provider type for the desktop."
    ),
):
    from agentdesk.vm import DesktopVM
    from agentdesk.vm.load import load_provider

    from surfkit.util import convert_unix_to_datetime

    provider_is_refreshed = {}
    vms = DesktopVM.find()
    if not vms:
        print("No desktops found")
    else:
        table = []
        for desktop in vms:
            if not desktop.provider:
                continue
            if provider:
                if desktop.provider.type != provider:
                    continue
            _provider = load_provider(desktop.provider)

            if not provider_is_refreshed.get(desktop.provider.type):
                if not desktop.reserved_ip:
                    _provider.refresh(log=False)
                    provider_is_refreshed[desktop.provider.type] = True
                    desktop = DesktopVM.get(desktop.name)
                    if not desktop:
                        continue

            table.append(
                [
                    desktop.name,
                    "desktop",
                    desktop.addr,
                    desktop.ssh_port,
                    desktop.status,
                    convert_unix_to_datetime(int(desktop.created)),
                    desktop.provider.type,  # type: ignore
                    desktop.reserved_ip,
                ]
            )

        print(
            tabulate(
                table,
                headers=[
                    "Name",
                    "Type",
                    "Address",
                    "SSH Port",
                    "Status",
                    "Created",
                    "Provider",
                    "Reserved IP",
                ],
            )
        )
        print("")


@list_group.command("trackers")
def list_trackers():
    from taskara.runtime.base import Tracker

    trackers = Tracker.find()

    if not trackers:
        print("No trackers found")
    else:
        table = []
        for server in trackers:
            table.append(
                [
                    server.name,
                    server.runtime.name(),
                    server.port,
                    server.status,
                ]
            )

        print(
            tabulate(
                table,
                headers=[
                    "Name",
                    "Runtime",
                    "Port",
                    "Status",
                ],
            )
        )
        print("")


@list_group.command("types")
def list_types():
    from surfkit.config import HUB_API_URL
    from surfkit.types import AgentType

    types = AgentType.find(remote=HUB_API_URL)
    if not types:
        raise ValueError(f"Agent type '{type}' not found")

    table = []
    for typ in types:
        table.append(
            [
                typ.name,
                typ.kind,
                typ.description,
            ]
        )

    print(
        tabulate(
            table,
            headers=[
                "Name",
                "Kind",
                "Description",
            ],
        )
    )
    print("")


@list_group.command("tasks")
def list_tasks(
    remote: bool = typer.Option(True, "--remote", "-r", help="List tasks from remote")
):
    import os
    from typing import List

    from taskara import Task

    from surfkit.config import HUB_API_URL, GlobalConfig
    from surfkit.env import HUB_API_KEY_ENV

    config = GlobalConfig.read()
    if not config.api_key:
        raise ValueError("No API key found. Please run `surfkit login` first.")

    os.environ[HUB_API_KEY_ENV] = config.api_key

    all_tasks: List[Task] = []

    if remote:
        try:
            tasks = Task.find(remote=HUB_API_URL)
            all_tasks.extend(tasks)
        except:
            pass

    try:
        tasks = Task.find()
        all_tasks.extend(tasks)
    except:
        pass

    table = []
    for task in all_tasks:
        table.append(
            [
                task.id,
                task.description,
                task.status,
            ]
        )

    print(
        tabulate(
            table,
            headers=[
                "ID",
                "Description",
                "Status",
            ],
        )
    )
    print("")


# 'get' sub-commands
@get_group.command("agent")
def get_agent(
    name: str = typer.Option(
        ..., "--name", "-n", help="The name of the agent to retrieve."
    ),
    runtime: Optional[str] = typer.Option(
        None, "--runtime", "-r", help="Get agent directly from the runtime"
    ),
):
    if runtime:
        if runtime == "docker":
            from surfkit.runtime.agent.docker import (
                DockerAgentRuntime,
                DockerConnectConfig,
            )

            dconf = DockerConnectConfig()
            runt = DockerAgentRuntime.connect(cfg=dconf)

        elif runtime == "kube":
            from surfkit.runtime.agent.kube import KubeAgentRuntime, KubeConnectConfig

            kconf = KubeConnectConfig()
            runt = KubeAgentRuntime.connect(cfg=kconf)

        elif runtime == "process":
            from surfkit.runtime.agent.process import (
                ProcessAgentRuntime,
                ProcessConnectConfig,
            )

            pconf = ProcessConnectConfig()
            runt = ProcessAgentRuntime.connect(cfg=pconf)

        else:
            raise ValueError(f"Unknown runtime '{runtime}'")

        instance = runt.get(name)
    else:
        from surfkit.runtime.agent.base import AgentInstance

        instances = AgentInstance.find(name=name)
        if not instances:
            raise ValueError(f"Agent instance '{name}' not found")
        instance = instances[0]

    rich.print_json(instance.to_v1().model_dump_json())


@get_group.command("device")
def get_device(
    name: str = typer.Option(
        ...,
        "--name",
        "-n",
        help="The name of the desktop to retrieve.",
    ),
    provider: Optional[str] = typer.Option(
        None, "--provider", "-p", help="The provider type for the desktop."
    ),
):
    from agentdesk.vm import DesktopVM
    from agentdesk.vm.load import load_provider

    if name:
        desktop = DesktopVM.get(name)
        if not desktop:
            raise ValueError("desktop not found")
        if not desktop.provider:
            raise ValueError("no desktop provider")
        if provider and desktop.provider.type != provider:
            print(f"Desktop '{name}' not found")
            return

        _provider = load_provider(desktop.provider)
        if not desktop.reserved_ip:
            _provider.refresh(log=False)
            desktop = DesktopVM.get(name)
            if not desktop:
                print(f"Desktop '{name}' not found")
                return

        if desktop:
            rich.print_json(desktop.to_v1_schema().model_dump_json())
        else:
            print(f"Desktop '{name}' not found")
        return


@get_group.command("type")
def get_type(
    name: str = typer.Option(
        ..., "--name", "-n", help="The name of the type to retrieve."
    )
):
    from surfkit.types import AgentType

    typer.echo(f"Getting type: {name}")
    from surfkit.config import HUB_API_URL

    types = AgentType.find(remote=HUB_API_URL, name=name)
    if not types:
        raise ValueError(f"Agent type '{type}' not found")
    agent_type = types[0]

    rich.print_json(agent_type.to_v1().model_dump_json())


@get_group.command("task")
def get_task(
    id: str = typer.Option(..., help="ID of the task"),
    remote: str = typer.Option(
        None,
        "--remote",
        "-r",
        help="Use a remote taskara instance, defaults to local db",
    ),
):
    import os
    from typing import List

    from taskara import Task

    from surfkit.config import HUB_API_URL, GlobalConfig
    from surfkit.env import HUB_API_KEY_ENV

    config = GlobalConfig.read()
    if not config.api_key:
        raise ValueError("No API key found. Please run `surfkit login` first.")

    os.environ[HUB_API_KEY_ENV] = config.api_key

    all_tasks: List[Task] = []

    if remote:
        try:
            tasks = Task.find(remote=HUB_API_URL, id=id)
            all_tasks.extend(tasks)
        except:
            pass

    try:
        tasks = Task.find(id=id)
        all_tasks.extend(tasks)
    except:
        pass

    if not all_tasks:
        raise ValueError(f"Task with ID '{id}' not found")
    task = all_tasks[0]
    rich.print_json(task.to_v1().model_dump_json())


# 'delete' sub-commands
@delete_group.command("agent")
def delete_agent(
    name: str = typer.Option(
        ..., "--name", "-n", help="The name of the agent to retrieve."
    ),
    runtime: Optional[str] = typer.Option(
        None,
        "--runtime",
        "-r",
        help="Delete the agent directly from the runtime. Options are 'docker', 'kube', 'process'.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Force delete the agent.",
    ),
):
    if runtime:
        if runtime == "docker":
            from surfkit.runtime.agent.docker import (
                DockerAgentRuntime,
                DockerConnectConfig,
            )

            dconf = DockerConnectConfig()
            runt = DockerAgentRuntime.connect(cfg=dconf)

        elif runtime == "kube":
            from surfkit.runtime.agent.kube import KubeAgentRuntime, KubeConnectConfig

            kconf = KubeConnectConfig()
            runt = KubeAgentRuntime.connect(cfg=kconf)

        elif runtime == "process":
            from surfkit.runtime.agent.process import (
                ProcessAgentRuntime,
                ProcessConnectConfig,
            )

            pconf = ProcessConnectConfig()
            runt = ProcessAgentRuntime.connect(cfg=pconf)

        else:
            raise ValueError(f"Unknown runtime '{runtime}'")

        runt.delete(name)
        typer.echo(f"Agent '{name}' deleted")

    else:
        from surfkit.runtime.agent.base import AgentInstance

        agents = AgentInstance.find(name=name)
        if not agents:
            raise ValueError(f"Agent '{name}' not found")
        agent = agents[0]
        agent.delete(force=force)
        typer.echo(f"Agent '{name}' deleted")


@delete_group.command("device")
def delete_device(
    name: str = typer.Option(
        ...,
        "--name",
        "-n",
        help="The name of the desktop to delete.",
    ),
    provider: Optional[str] = typer.Option(
        None, "--provider", "-p", help="The provider type for the desktop."
    ),
):
    from agentdesk.vm import DesktopVM
    from agentdesk.vm.load import load_provider

    desktop = DesktopVM.get(name)
    if not desktop:
        raise ValueError("desktop not found")
    if not desktop.provider:
        raise ValueError("no desktop provider")
    if provider and desktop.provider.type != provider:
        print(f"Desktop '{name}' not found")
        return

    _provider = load_provider(desktop.provider)
    if not desktop.reserved_ip:
        _provider.refresh(log=False)
        desktop = DesktopVM.get(name)
        if not desktop:
            print(f"Desktop '{name}' not found")
            return

    if desktop:
        _provider.delete(name)
        typer.echo("Desktop deleted")
    else:
        print(f"Desktop '{name}' not found")
    return


@delete_group.command("tracker")
def delete_tracker(
    name: str = typer.Option(
        ...,
        "--name",
        "-n",
        help="The name of the tracker to delete",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Whether to force delete the tracker",
    ),
):
    from taskara.runtime.base import Tracker

    trackers = Tracker.find(name=name)
    if not trackers:
        raise ValueError(f"Tracker '{name}' not found")

    tracker = trackers[0]

    tracker.delete(force=force)
    typer.echo("Tracker deleted")
    return


@delete_group.command("type")
def delete_type(
    name: str = typer.Option(..., "--name", "-n", help="The name of the type."),
):
    from surfkit.config import HUB_API_URL
    from surfkit.types import AgentType

    types = AgentType.find(remote=HUB_API_URL, name=name)
    if not types:
        raise ValueError(f"Agent type '{type}' not found")
    agent_type = types[0]
    agent_type.remove()
    typer.echo(f"Agent type '{name}' deleted")


# View subcommands


@view_group.command("device")
def view_device(
    name: str = typer.Option(
        ..., "--name", "-n", help="The name of the device to view."
    ),
    background: bool = typer.Option(
        False, "--background", "-b", help="Run the viewer in background mode"
    ),
):
    from agentdesk.vm import DesktopVM

    desktop = DesktopVM.get(name)
    if not desktop:
        raise ValueError(f"Desktop '{name}' not found")

    desktop.view(background=background)


# Other commands
@app.command(help="Login to the hub")
def login():
    from surfkit.config import HUB_URL, GlobalConfig

    url = urljoin(HUB_URL, "cli-login")
    typer.echo(f"\nVisit {url} to get an API key\n")

    webbrowser.open(url)
    api_key = typer.prompt("Enter your API key", hide_input=True)

    config = GlobalConfig.read()
    config.api_key = api_key
    config.write()

    typer.echo("\nLogin successful!")


@app.command(help="Publish an agent")
def publish(
    agent_file: str = typer.Option(
        "./agent.yaml", "--agent-file", "-f", help="Agent file to use"
    ),
    build: bool = typer.Option(True, "--build", "-b", help="Build the docker image"),
    version: str = typer.Option("latest", "--version", "-v", help="Version to build"),
):
    from surfkit.types import AgentType

    typ = AgentType.from_file(agent_file)

    if build:
        from .util import build_docker_image

        typer.echo(
            f"Building docker image for agent '{agent_file}' version '{version}'"
        )

        if not typ.versions:
            raise ValueError(f"No versions found for agent {typ.name}")

        ver = typ.versions.get(version)
        if not ver:
            raise ValueError(f"Version {version} not found in {typ.name}")

        build_docker_image(dockerfile_path="./Dockerfile", tag=ver, push=True)

    from surfkit.config import HUB_API_URL, GlobalConfig

    url = urljoin(HUB_API_URL, "v1/agenttypes")
    typer.echo(f"\nPublishing agent to {url}...\n")

    config = GlobalConfig.read()
    if not config.api_key:
        raise ValueError("No API key found. Please run `surfkit login` first.")

    headers = {"Authorization": f"Bearer {config.api_key}"}
    resp = requests.post(url, json=typ.to_v1().model_dump(), headers=headers)
    resp.raise_for_status()
    typer.echo(f"Agent published!")


@app.command(help="Create a new agent repo")
def new(
    template: str = typer.Option(
        "surf4v",
        "--template",
        "-t",
        help="Template to use. Options are 'surf4v' or 'surfskelly'",
    )
):
    from rich.prompt import Prompt

    from .new import new_agent
    from .util import get_git_global_user_config

    name = Prompt.ask("Enter agent name")
    if not name.isalnum() or len(name) > 50:
        typer.echo(
            "Invalid agent name. Must be alphanumeric and less than 50 characters."
        )

        name = Prompt.ask("Enter agent name")
        if not name.isalnum() or len(name) > 50:
            typer.echo(
                "Invalid agent name. Must be alphanumeric and less than 50 characters."
            )
            raise typer.Exit()

    description = Prompt.ask("Describe the agent")
    git_user_ref = Prompt.ask(
        "Enter git user reference", default=get_git_global_user_config()
    )
    image_repo = Prompt.ask("Enter docker image repo")
    icon_url = Prompt.ask(
        "Enter icon url",
        default="https://tinyurl.com/y5u4u7te",
    )
    new_agent(
        name=name,
        description=description,
        git_user_ref=git_user_ref,
        img_repo=image_repo,
        icon_url=icon_url,
        template=template,
    )


@app.command(help="Build the agent container")
def build(
    version: str = typer.Option("latest", "--version", "-v", help="Version to build"),
    agent_file: str = typer.Option(
        "./agent.yaml", "--agent-file", "-f", help="Agent file to use"
    ),
    push: bool = typer.Option(True, "--push", "-p", help="Also push the image"),
):
    from surfkit.types import AgentType

    from .util import build_docker_image

    typer.echo(f"Building docker image for agent '{agent_file}' version '{version}'")

    typ = AgentType.from_file(agent_file)
    if not typ.versions:
        raise ValueError(f"No versions found for agent {typ.name}")

    ver = typ.versions.get(version)
    if not ver:
        raise ValueError(f"Version {version} not found in {typ.name}")

    build_docker_image("./Dockerfile", ver, push)


@app.command(help="Use an agent to solve a task")
def solve(
    description: str = typer.Option(
        ..., "--description", "-d", help="Description of the task."
    ),
    agent: Optional[str] = typer.Option(
        None, "--agent", "-a", help="Name of the agent to use."
    ),
    agent_runtime: Optional[str] = typer.Option(
        None, "--agent-runtime", "-r", help="Runtime environment for the agent."
    ),
    agent_type: Optional[str] = typer.Option(
        None, "--agent-type", "-t", help="Type of agent to use."
    ),
    agent_file: Optional[str] = typer.Option(
        None, "--agent-file", "-f", help="Path to agent config file."
    ),
    agent_version: Optional[str] = typer.Option(None, help="Version of agent to use."),
    device: Optional[str] = typer.Option(
        None, help="Name of device to use if applicable."
    ),
    device_type: Optional[str] = typer.Option(
        None, help="Name of the type of device if using one."
    ),
    device_provider: Optional[str] = typer.Option(
        None, "--device-provider", "-p", help="The provider type for the device."
    ),
    tracker: Optional[str] = typer.Option(None, help="Name of tracker to use."),
    tracker_runtime: Optional[str] = typer.Option(
        None,
        help="Runtime to create a tracker if needed. Options are 'docker' or 'kube'.",
    ),
    tracker_remote: Optional[str] = typer.Option(
        None, help="URL of remote tracker if needed."
    ),
    max_steps: int = typer.Option(30, help="Maximum steps for the task."),
    kill: bool = typer.Option(
        False, "--kill", "-k", help="Whether to kill the agent when done"
    ),
    view: bool = typer.Option(True, "--view", "-v", help="Whether to view the device"),
    follow: bool = typer.Option(True, help="Whether to follow the agent logs"),
    starting_url: Optional[str] = typer.Option(
        None, help="Starting URL if applicable."
    ),
):
    from taskara import Task
    from taskara.runtime.base import Tracker
    from agentdesk import Desktop
    from taskara import Task

    from surfkit.server.models import V1SolveTask
    from surfkit.util import find_open_port
    from surfkit.types import AgentType

    if tracker:
        trackers = Tracker.find(name=tracker)
        if not trackers:
            raise ValueError(f"Expected tracker with name '{tracker}'")
        task_svr = trackers[0]

        local_port = task_svr.port
        if task_svr.runtime.requires_proxy():
            local_port = find_open_port(9070, 10070)
            if not local_port:
                raise SystemError("No available ports found")
            task_svr.proxy(local_port=local_port)
        _remote_task_svr = f"http://localhost:{local_port}"

    elif tracker_runtime:

        if tracker_runtime == "docker":
            from taskara.runtime.docker import DockerTrackerRuntime

            task_runt = DockerTrackerRuntime()

        elif tracker_runtime == "kube":
            from taskara.runtime.kube import KubeTrackerRuntime

            task_runt = KubeTrackerRuntime()

        else:
            typer.echo(f"Invalid runtime: {tracker_runtime}")
            raise typer.Exit()

        name = get_random_name(sep="-")
        if not name:
            raise SystemError("Name is required for tracker")

        task_svr = task_runt.run(name=name, auth_enabled=False)
        typer.echo(f"Tracker '{name}' created using '{tracker_runtime}' runtime")

        local_port = task_svr.port
        if task_svr.runtime.requires_proxy():
            local_port = find_open_port(9070, 10070)
            if not local_port:
                raise SystemError("No available ports found")
            task_svr.proxy(local_port=local_port)
        _remote_task_svr = f"http://localhost:{local_port}"

    elif tracker_remote:
        _remote_task_svr = tracker_remote

    else:
        trackers = Tracker.find()
        if not trackers:
            create = typer.confirm("No trackers found. Would you like to create one?")
            if create:
                from taskara.runtime.docker import DockerTrackerRuntime

                task_runt = DockerTrackerRuntime()

                name = get_random_name(sep="-")
                if not name:
                    raise SystemError("Name is required for tracker")

                task_svr = task_runt.run(name=name, auth_enabled=False)
                typer.echo(
                    f"Tracker '{name}' created using '{task_runt.name()}' runtime"
                )
            else:
                raise ValueError(
                    "`tracker`, `tracker_runtime`, or `tracker_remote` flag must be provided. Or a tracker must be running"
                )
        else:
            task_svr = trackers[0]
            typer.echo(
                f"Using tracker '{task_svr.name}' running on '{task_svr.runtime.name()}'"
            )

        local_port = task_svr.port
        if task_svr.runtime.requires_proxy():
            local_port = find_open_port(9070, 10070)
            if not local_port:
                raise SystemError("No available ports found")
            task_svr.proxy(local_port=local_port)
        _remote_task_svr = f"http://localhost:{local_port}"

    if not agent_runtime:
        if agent_file:
            agent_runtime = "process"
        elif agent_type:
            agent_runtime = "docker"
        else:
            agent_runtime = "docker"

    runt = None
    if agent:
        instances = AgentInstance.find(name=agent)
        if not instances:
            raise ValueError(f"Expected instances of '{agent}'")
        typer.echo(f"Found agent instance '{agent}'")
        instance = instances[0]
        runt = instance.runtime

    else:
        if agent_runtime == "docker":
            from surfkit.runtime.agent.docker import (
                DockerAgentRuntime,
                DockerConnectConfig,
            )

            dconf = DockerConnectConfig()
            runt = DockerAgentRuntime.connect(cfg=dconf)

        elif agent_runtime == "kube":
            from surfkit.runtime.agent.kube import KubeAgentRuntime, KubeConnectConfig

            kconf = KubeConnectConfig()
            runt = KubeAgentRuntime.connect(cfg=kconf)

        elif agent_runtime == "process":
            from surfkit.runtime.agent.process import (
                ProcessAgentRuntime,
                ProcessConnectConfig,
            )

            pconf = ProcessConnectConfig()
            runt = ProcessAgentRuntime.connect(cfg=pconf)

        else:
            if agent_file or agent_type:
                raise ValueError(f"Unknown runtime '{agent_runtime}'")

    if not runt:
        raise ValueError(f"Unknown runtime '{agent_runtime}'")

    v1device = None
    _device = None
    if device_type:
        if device_type == "desktop":
            from agentdesk.server.models import V1ProviderData
            from agentdesk.vm.load import load_provider

            data = V1ProviderData(type=device_provider)
            _provider = load_provider(data)

            typer.echo(f"Creating desktop '{agent}' using '{device_provider}' provider")
            try:
                vm = _provider.create(
                    name=agent,
                )
                _device = Desktop.from_vm(vm)
                v1device = _device.to_v1()
            except KeyboardInterrupt:
                print("Keyboard interrupt received, exiting...")
                return
        else:
            raise ValueError(f"unknown device type {device_type}")

    vm = None
    if device:
        typer.echo(f"finding device '{device}'...")
        vms = Desktop.find(name=device)
        if not vms:
            raise ValueError(f"Device '{device}' not found")
        vm = vms[0]
        _device = Desktop.from_vm(vm)
        v1device = _device.to_v1()
        typer.echo(f"found device '{device}'...")

    if agent_type:
        from surfkit.config import HUB_API_URL

        types = AgentType.find(remote=HUB_API_URL, name=agent_type)
        if not types:
            raise ValueError(f"Agent type '{agent_type}' not found")
        typ = types[0]
        if not agent:
            agent = get_random_name("-")
            if not agent:
                raise ValueError("could not generate agent name")
        typer.echo(f"creating agent {agent}...")
        instance = runt.run(agent_type=typ, name=agent, version=agent_version)
        agent = instance.name

    if agent_file:
        typ = AgentType.from_file(agent_file)
        if not agent:
            from surfkit.runtime.agent.util import instance_name

            agent = instance_name(typ)

        types = AgentType.find(name=typ.name)
        if types:
            typ = types[0]
            typ.update(typ.to_v1())
        else:
            typ.save()

        typer.echo(f"creating agent {agent} from file {agent_file}...")
        instance = runt.run(agent_type=typ, name=agent, version=agent_version)
        agent = instance.name

    if not agent:
        raise ValueError("Either agent or agent_type needs to be provided")

    if _device and view:
        typer.echo("viewing device...")
        from surfkit.cli.view import view as _view

        instances = AgentInstance.find(name=agent)
        if not instances:
            raise ValueError(f"agent '{agent}' not found")
        instance = instances[0]

        if not vm:
            raise ValueError("vm not found for ui")

        _view(desk=vm, agent=instance, tracker_addr=_remote_task_svr, background=True)

    params = {}
    if starting_url:
        params["site"] = starting_url

    task = Task(
        description=description,
        parameters=params,
        max_steps=max_steps,
        device=v1device,
        assigned_to=agent,
        assigned_type=agent_type,
        remote=_remote_task_svr,
    )

    typer.echo(f"Solving task '{task.description}' with agent '{agent}'...")
    solve_v1 = V1SolveTask(task=task.to_v1())
    runt.solve_task(agent, solve_v1, follow_logs=follow, attach=kill)

    if kill and not follow:
        typer.echo(f"Killing agent {agent}...")
        runt.delete(agent)


@app.command("logs")
def get_logs(
    name: str = typer.Option(
        ...,
        "--name",
        "-n",
        help="The name of the agent whose logs are to be retrieved.",
    ),
    runtime: Optional[str] = typer.Option(
        None, "--runtime", "-r", help="The runtime of the agent."
    ),
    follow: bool = typer.Option(
        False, "--follow", "-f", help="Whether to continuously follow the logs."
    ),
):
    """
    Retrieve agent logs
    """
    if runtime:
        if runtime == "docker":
            from surfkit.runtime.agent.docker import (
                DockerAgentRuntime,
                DockerConnectConfig,
            )

            config = DockerConnectConfig()
            runtime_instance = DockerAgentRuntime.connect(cfg=config)

        elif runtime == "kube":
            from surfkit.runtime.agent.kube import KubeAgentRuntime, KubeConnectConfig

            config = KubeConnectConfig()
            runtime_instance = KubeAgentRuntime.connect(cfg=config)

        elif runtime == "process":
            from surfkit.runtime.agent.process import (
                ProcessAgentRuntime,
                ProcessConnectConfig,
            )

            config = ProcessConnectConfig()
            runtime_instance = ProcessAgentRuntime.connect(cfg=config)

        else:
            typer.echo(f"Unsupported runtime: {runtime}")
            raise typer.Exit(1)

        # Fetch logs using the AgentRuntime instance
        try:
            logs = runtime_instance.logs(name, follow)
            if isinstance(logs, str):
                typer.echo(logs)
            else:
                # Handle log streaming
                try:
                    for log_entry in logs:
                        typer.echo(log_entry)
                        if not follow:
                            break
                except KeyboardInterrupt:
                    typer.echo("Stopped following logs.")
        except Exception as e:
            typer.echo(f"Failed to retrieve logs: {str(e)}")

    else:
        from surfkit.runtime.agent.base import AgentInstance

        instances = AgentInstance.find(name=name)
        if not instances:
            typer.echo(f"Agent '{name}' not found")
            raise typer.Exit(1)
        instance = instances[0]
        logs = instance.logs(follow=follow)
        if isinstance(logs, str):
            typer.echo(logs)
        else:
            # Handle log streaming
            try:
                for log_entry in logs:
                    typer.echo(log_entry)
                    if not follow:
                        break
            except KeyboardInterrupt:
                typer.echo("Stopped following logs.")


if __name__ == "__main__":
    app()
