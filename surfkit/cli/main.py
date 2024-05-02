from logging import config
from tkinter.font import names
from urllib.parse import urljoin
from typing import Optional
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as pkgversion
from click import Option
import rich
import yaml

import typer
import webbrowser
from namesgenerator import get_random_name
import requests
from tabulate import tabulate

art = """
 _______                ___  __  __  __  __   
|     __|.--.--..----..'  _||  |/  ||__||  |_ 
|__     ||  |  ||   _||   _||     < |  ||   _|
|_______||_____||__|  |__|  |__|\__||__||____|
"""

app = typer.Typer()

# Sub-command groups
create = typer.Typer(help="Create an agent or device")
list_group = typer.Typer(help="List resources")
get = typer.Typer(help="Get resources")
view = typer.Typer(help="View resources")

app.add_typer(create, name="create")
app.add_typer(list_group, name="list")
app.add_typer(get, name="get")
app.add_typer(view, name="view")


# Callback for showing help
def show_help(ctx: typer.Context, command_group: str):
    if ctx.invoked_subcommand is None:
        if command_group == "root":
            typer.echo(art)
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
@create.callback(invoke_without_command=True)
def create_default(ctx: typer.Context):
    show_help(ctx, "create")


# 'list' command group callback
@list_group.callback(invoke_without_command=True)
def list_default(ctx: typer.Context):
    show_help(ctx, "list")


# 'get' command group callback
@get.callback(invoke_without_command=True)
def get_default(ctx: typer.Context):
    show_help(ctx, "get")


# 'create' sub-commands
@create.command("device")
def create_device(
    name: Optional[str] = typer.Option(
        None, help="The name of the desktop to create. Defaults to a generated name."
    ),
    provider: str = typer.Option(
        "qemu",
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

    if not name:
        name = get_random_name(sep="-")

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


@create.command("agent")
def create_agent(
    runtime: str = typer.Option("docker", help="The runtime to use."),
    file: str = typer.Option(
        "./agent.yaml", help="Path to the agent configuration file."
    ),
    name: Optional[str] = typer.Option(
        None, help="Name of the agent. Defaults to a generated name."
    ),
    type: Optional[str] = typer.Option(None, help="Type of the agent if predefined."),
):
    if not name:
        name = get_random_name(sep="-")
        if not name:
            raise ValueError("could not generate name")
    typer.echo(f"Running agent '{file}' with runtime '{runtime}'...")

    from surfkit.models import V1AgentType
    from surfkit.types import AgentType

    if runtime == "docker":
        from surfkit.runtime.agent.docker import (
            DockerAgentRuntime,
            ConnectConfig as DockerConnectConfig,
        )

        conf = DockerConnectConfig()
        runt = DockerAgentRuntime.connect(cfg=conf)

    elif runtime == "kube":
        from surfkit.runtime.agent.kube import (
            KubernetesAgentRuntime,
            ConnectConfig as KubeConnectConfig,
        )

        conf = KubeConnectConfig()
        runt = KubernetesAgentRuntime.connect(cfg=conf)

    elif runtime == "process":
        from surfkit.runtime.agent.process import (
            ProcessAgentRuntime,
            ConnectConfig as ProcessConnectConfig,
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
                agent_type_model = V1AgentType(**data)
        except Exception as e:
            raise ValueError(f"Failed to parse find or parse agent.yaml: {e}")

        agent_type = AgentType.from_v1(agent_type_model)

    runt.run(agent_type, name)
    typer.echo(f"Successfully created agent")


# 'list' sub-commands


@list_group.command("agents")
def list_agents(runtime: str = "docker"):
    if runtime == "docker":
        from surfkit.runtime.agent.docker import (
            DockerAgentRuntime,
            ConnectConfig as DockerConnectConfig,
        )

        dconf = DockerConnectConfig()
        runt = DockerAgentRuntime.connect(cfg=dconf)

    elif runtime == "kube":
        from surfkit.runtime.agent.kube import (
            KubernetesAgentRuntime,
            ConnectConfig as KubeConnectConfig,
        )

        kconf = KubeConnectConfig()
        runt = KubernetesAgentRuntime.connect(cfg=kconf)

    elif runtime == "process":
        from surfkit.runtime.agent.process import (
            ProcessAgentRuntime,
            ConnectConfig as ProcessConnectConfig,
        )

        pconf = ProcessConnectConfig()
        runt = ProcessAgentRuntime.connect(cfg=pconf)

    else:
        raise ValueError(f"Unknown runtime '{runtime}'")

    agents = runt.list()
    table = []
    for name in agents:
        table.append([name, runtime])

    print(
        tabulate(
            table,
            headers=[
                "Name",
                "Runtime",
                "Type",
            ],
        )
    )
    print("")


@list_group.command("devices")
def list_devices(
    provider: Optional[str] = typer.Option(
        None, help="The provider type for the desktop."
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


@list_group.command("types")
def list_types():
    from surfkit.types import AgentType
    from surfkit.config import HUB_API_URL

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


# 'get' sub-commands
@get.command("agent")
def get_agent(
    name: str = typer.Option(..., help="The name of the agent to retrieve."),
    runtime: str = typer.Option("docker", help="The runtime of the agent to retrieve."),
):
    if runtime == "docker":
        from surfkit.runtime.agent.docker import (
            DockerAgentRuntime,
            ConnectConfig as DockerConnectConfig,
        )

        dconf = DockerConnectConfig()
        runt = DockerAgentRuntime.connect(cfg=dconf)

    elif runtime == "kube":
        from surfkit.runtime.agent.kube import (
            KubernetesAgentRuntime,
            ConnectConfig as KubeConnectConfig,
        )

        kconf = KubeConnectConfig()
        runt = KubernetesAgentRuntime.connect(cfg=kconf)

    elif runtime == "process":
        from surfkit.runtime.agent.process import (
            ProcessAgentRuntime,
            ConnectConfig as ProcessConnectConfig,
        )

        pconf = ProcessConnectConfig()
        runt = ProcessAgentRuntime.connect(cfg=pconf)

    else:
        raise ValueError(f"Unknown runtime '{runtime}'")

    instance = runt.get(name)
    rich.print_json(instance.to_v1().model_dump_json())


@get.command("device")
def get_device(
    name: str = typer.Option(
        help="The name of the desktop to retrieve.",
    ),
    provider: Optional[str] = typer.Option(
        None, help="The provider type for the desktop."
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


@get.command("type")
def get_type(name: str):
    from surfkit.types import AgentType

    typer.echo(f"Getting type: {name}")
    from surfkit.config import HUB_API_URL

    types = AgentType.find(remote=HUB_API_URL, name=name)
    if not types:
        raise ValueError(f"Agent type '{type}' not found")
    agent_type = types[0]
    rich.print_json(agent_type.to_v1().model_dump_json())


# Other commands
@app.command(help="Login to the hub")
def login():
    from surfkit.config import GlobalConfig, HUB_URL

    url = urljoin(HUB_URL, "cli-login")
    typer.echo(f"\nVisit {url} to get an API key\n")

    webbrowser.open(url)
    api_key = typer.prompt("Enter your API key", hide_input=True)

    config = GlobalConfig.read()
    config.api_key = api_key
    config.write()

    typer.echo("\nLogin successful!")


@app.command(help="Publish an agent")
def publish(path: str = "./agent.yaml"):
    from surfkit.config import GlobalConfig, HUB_API_URL

    url = urljoin(HUB_API_URL, "v1/agenttypes")
    typer.echo(f"\nPublishing agent to {url}...\n")

    from surfkit.models import V1AgentType

    with open(path, "r") as f:
        agent_type = V1AgentType.model_validate(yaml.safe_load(f))

    config = GlobalConfig.read()
    if not config.api_key:
        raise ValueError("No API key found. Please run `surfkit login` first.")

    headers = {"Authorization": f"Bearer {config.api_key}"}
    resp = requests.post(url, json=agent_type.model_dump(), headers=headers)
    resp.raise_for_status()
    typer.echo(f"Agent published!")


@app.command(help="Create a new agent repo")
def new():
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
    )


@app.command(help="Use an agent to solve a task")
def solve(
    description: str = typer.Option(..., help="Description of the task."),
    agent_name: Optional[str] = typer.Option(None, help="Name of the agent to use."),
    agent_type: Optional[str] = typer.Option(None, help="Type of agent to use."),
    agent_file: Optional[str] = typer.Option(None, help="Path to agent config file."),
    agent_version: Optional[str] = typer.Option(None, help="Version of agent to use."),
    device: Optional[str] = typer.Option(
        None, help="Name of device to use if applicable."
    ),
    device_type: Optional[str] = typer.Option(
        None, help="Name of the type of device if using one."
    ),
    device_provider: Optional[str] = typer.Option(
        None, help="The provider type for the device."
    ),
    max_steps: int = typer.Option(20, help="Maximum steps for the task."),
    starting_url: Optional[str] = typer.Option(
        None, help="Starting URL if applicable."
    ),
    runtime: str = typer.Option("docker", help="Runtime environment for the agent."),
    kill: bool = typer.Option(False, help="Whether to kill the agent when done"),
    view: bool = typer.Option(True, help="Whether to view the device"),
    follow: bool = typer.Option(True, help="Whether to follow the agent logs"),
):
    from taskara import Task
    from agentdesk import Desktop
    from surfkit.types import AgentType

    if runtime == "docker":
        from surfkit.runtime.agent.docker import (
            DockerAgentRuntime,
            ConnectConfig as DockerConnectConfig,
        )

        dconf = DockerConnectConfig()
        runt = DockerAgentRuntime.connect(cfg=dconf)

    elif runtime == "kube":
        from surfkit.runtime.agent.kube import (
            KubernetesAgentRuntime,
            ConnectConfig as KubeConnectConfig,
        )

        kconf = KubeConnectConfig()
        runt = KubernetesAgentRuntime.connect(cfg=kconf)

    elif runtime == "process":
        from surfkit.runtime.agent.process import (
            ProcessAgentRuntime,
            ConnectConfig as ProcessConnectConfig,
        )

        pconf = ProcessConnectConfig()
        runt = ProcessAgentRuntime.connect(cfg=pconf)

    else:
        raise ValueError(f"Unknown runtime '{runtime}'")

    v1device = None
    _device = None
    if device_type:
        if device_type == "desktop":
            from agentdesk.server.models import V1ProviderData
            from agentdesk.vm.load import load_provider

            data = V1ProviderData(type=device_provider)
            _provider = load_provider(data)

            typer.echo(
                f"Creating desktop '{agent_name}' using '{device_provider}' provider"
            )
            try:
                vm = _provider.create(
                    name=agent_name,
                )
                _device = Desktop.from_vm(vm)
                v1device = _device.to_v1()
            except KeyboardInterrupt:
                print("Keyboard interrupt received, exiting...")
                return
        else:
            raise ValueError(f"unknown device type {device_type}")

    if device:
        typer.echo(f"finding device '{device}'...")
        vms = Desktop.find(name=device)
        if not vms:
            raise ValueError(f"Device '{device}' not found")
        vm = vms[0]
        _device = Desktop.from_vm(vm)
        v1device = _device.to_v1()
        typer.echo(f"found device '{device}'...")

    if _device and view:
        typer.echo("viewing device...")
        _device.view(True)

    if agent_type:
        from surfkit.config import HUB_API_URL

        types = AgentType.find(remote=HUB_API_URL, name=agent_type)
        if not types:
            raise ValueError(f"Agent type '{agent_type}' not found")
        typ = types[0]
        if not agent_name:
            agent_name = get_random_name("-")
            if not agent_name:
                raise ValueError("could not generate agent name")
        typer.echo(f"creating agent {agent_name}...")
        runt.run(agent_type=typ, name=agent_name, version=agent_version)

    if agent_file:
        typ = AgentType.from_file(agent_file)
        if not agent_name:
            agent_name = get_random_name("-")
            if not agent_name:
                raise ValueError("could not generate agent name")
        typer.echo(f"creating agent {agent_name} from file {agent_file}...")
        runt.run(agent_type=typ, name=agent_name, version=agent_version)

    if not agent_name:
        raise ValueError("Either agent_name or agent_type needs to be provided")

    task = Task(
        description=description,
        parameters={"site": starting_url},
        max_steps=max_steps,
        device=v1device,
        assigned_to=agent_name,
        assigned_type=agent_type,
    )

    typer.echo(f"Solving task '{task.description}' with agent {agent_name}...")
    runt.solve_task(agent_name, task.to_v1(), follow_logs=follow)

    if kill:
        typer.echo(f"Killing agent {agent_name}...")
        runt.delete(agent_name)


if __name__ == "__main__":
    app()
