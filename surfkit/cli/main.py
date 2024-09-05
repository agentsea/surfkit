import logging
import webbrowser
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as pkgversion
from typing import Optional
from urllib.parse import urljoin

import requests
import rich
import typer
import yaml
from namesgenerator import get_random_name
from tabulate import tabulate
from agentdesk.vm.ec2 import EC2Provider


logger = logging.getLogger(__name__)

app = typer.Typer()

# Sub-command groups
create_group = typer.Typer(help="Create resources")
list_group = typer.Typer(help="List resources")
get_group = typer.Typer(help="Get resources")
view_group = typer.Typer(help="View resources")
delete_group = typer.Typer(help="Delete resources")
logs_group = typer.Typer(help="Resource logs")
clean_group = typer.Typer(help="Clean resources")

app.add_typer(create_group, name="create")
app.add_typer(list_group, name="list")
app.add_typer(get_group, name="get")
app.add_typer(view_group, name="view")
app.add_typer(delete_group, name="delete")
app.add_typer(logs_group, name="logs")
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
def default(
    ctx: typer.Context,
    debug: bool = typer.Option(False, "--debug", "-d", help="Enable debug mode"),
):
    if debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
        raise typer.Exit()


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


# 'logs' command group callback
@logs_group.callback(invoke_without_command=True)
def logs_default(ctx: typer.Context):
    show_help(ctx, "logs")


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
    region: Optional[str] = typer.Option(
        "us-east-1",
        help=f"AWS region. Defaults to 'us-east-1'. Options: {', '.join(EC2Provider.AVAILABLE_REGIONS)}",
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
        data = V1ProviderData(type=provider, args={"region": region})
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
    auth_enabled: bool = typer.Option(
        False, "--auth-enabled", "-e", help="Whether to enable auth for the tracker."
    ),
):

    if runtime == "docker":
        from taskara.runtime.docker import DockerConnectConfig, DockerTrackerRuntime

        runt = DockerTrackerRuntime(DockerConnectConfig(image=image))

    elif runtime == "kube":
        from taskara.runtime.kube import KubeConnectConfig, KubeTrackerRuntime

        runt = KubeTrackerRuntime(KubeConnectConfig(image=image))

    else:
        typer.echo(f"Invalid runtime: {runtime}")
        raise typer.Exit()

    if not name:
        name = get_random_name(sep="-")
        if not name:
            raise SystemError("Name is required for tracker")

    server = runt.run(name=name, auth_enabled=auth_enabled)
    typer.echo(f"Tracker '{name}' created using '{runtime}' runtime")


@create_group.command("benchmark")
def create_benchmark(
    file: str = typer.Argument(
        help="The file to create the benchmark from.",
    ),
    tracker: str = typer.Option(
        None,
        "--tracker",
        "-t",
        help="The tracker to use for the benchmark",
    ),
):
    import yaml
    from taskara import V1Benchmark
    from taskara.runtime.base import Tracker

    if not tracker:
        trackers = Tracker.find()
        if not trackers:
            typer.echo("No trackers found")
            raise typer.Exit()
        trck = trackers[0]
    else:
        trackers = Tracker.find(name=tracker)
        if not trackers:
            typer.echo(f"Tracker '{tracker}' not found")
            raise typer.Exit()
        trck = trackers[0]

    with open(file) as f:
        dct = yaml.safe_load(f)
        v1_benchmark = V1Benchmark.model_validate(dct)

    status, text = trck.call(
        "/v1/benchmarks", method="POST", data=v1_benchmark.model_dump()
    )
    if status != 200:
        typer.echo(f"Error creating benchmark: {text}")
        raise typer.Exit()

    typer.echo(f"Benchmark '{v1_benchmark.name}' created using '{trck.name}' tracker")


@create_group.command("task")
def create_task(
    description: str = typer.Option(
        ..., "--description", "-d", help="A task description"
    ),
    assigned_to: Optional[str] = typer.Option(
        None, "--assigned-to", "-o", help="Agent to assign the task to"
    ),
    assigned_type: Optional[str] = typer.Option(
        None, "--assigned-type", "-t", help="Agent type to assign the task to"
    ),
    device: Optional[str] = typer.Option(
        None, "--device", "-e", help="Device to give the agent"
    ),
    device_type: Optional[str] = typer.Option(
        None, "--device-type", "-y", help="Device type to give the agen"
    ),
    max_steps: int = typer.Option(
        30, "--max-steps", "-m", help="Max steps the agent can take"
    ),
    tracker: Optional[str] = typer.Option(
        None, "--tracker", help="Tracker to use. Defaults to the hub."
    ),
):
    from agentdesk import Desktop
    from devicebay import V1DeviceType
    from taskara.runtime.base import Tracker
    from taskara.task import Task

    from surfkit.config import AGENTSEA_HUB_API_URL, GlobalConfig

    _device_type = None
    if device_type:
        _device_type = V1DeviceType(name=device_type)

    _device = None
    if device:
        desk_vms = Desktop.find(name=device)
        if not desk_vms:
            typer.echo(f"Desktop '{device}' not found")
            raise typer.Exit()
        _desk_vm = desk_vms[0]

        _desk = Desktop.from_vm(_desk_vm)
        _device = _desk.to_v1()

    if tracker:
        from surfkit.util import find_open_port

        trackers = Tracker.find(name=tracker)
        if not trackers:
            typer.echo(f"Tracker '{tracker}' not found")
            raise typer.Exit()
        trck = trackers[0]

        local_port = trck.port
        if trck.runtime.requires_proxy():
            local_port = find_open_port(9070, 10070)
            if not local_port:
                raise SystemError("No available ports found")
            trck.proxy(local_port=local_port)
        _remote_tracker = f"http://localhost:{local_port}"
    else:
        config = GlobalConfig.read()
        if not config.api_key:
            raise ValueError(
                "No API key found. Please run `surfkit login` first or provide a tracker"
            )
        _remote_tracker = AGENTSEA_HUB_API_URL

    task = Task(
        description=description,
        assigned_to=assigned_to,
        assigned_type=assigned_type,
        device=_device,
        device_type=_device_type,
        max_steps=max_steps,
        remote=_remote_tracker,
    )

    typer.echo(f"Task '{task.id}' created")


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
    auth_enabled: bool = typer.Option(
        False, "--auth-enabled", "-e", help="Whether to enable auth for the agent."
    ),
    local_keys: bool = typer.Option(
        False, "--local-keys", "-l", help="Use local API keys."
    ),
    debug: bool = typer.Option(False, help="Run the agent with debug logging"),
):

    from surfkit.server.models import V1AgentType
    from surfkit.types import AgentType

    from ..env_opts import find_envs

    if not runtime:
        if type:
            runtime = "docker"
        else:
            runtime = "process"

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

        from typing import List

        from surfkit.config import AGENTSEA_HUB_API_URL, GlobalConfig

        all_types: List[AgentType] = []

        type_parts = type.split("/")

        if len(type_parts) == 1:
            types = AgentType.find(name=type)
            if types:
                all_types.extend(types)
        elif len(type_parts) == 2:
            try:
                types = AgentType.find(
                    remote=AGENTSEA_HUB_API_URL,
                    namespace=type_parts[0],
                    name=type_parts[1],
                )
                if types:
                    all_types.extend(types)
            except Exception as e:
                logger.debug(f"Failed to load global config: {e}")

        if not all_types:
            raise ValueError(f"Agent type '{type}' not found")

        agent_type = all_types[0]
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

    env_vars = find_envs(agent_type, use_local=local_keys)
    if type:
        typer.echo(
            f"Running agent '{type}' with runtime '{runtime}' and name '{name}'..."
        )
    else:
        typer.echo(
            f"Running agent '{file}' with runtime '{runtime}' and name '{name}'..."
        )

    try:
        runt.run(
            agent_type,
            name,
            auth_enabled=auth_enabled,
            env_vars=env_vars,
            debug=debug,
        )
    except Exception as e:
        typer.echo(f"Failed to run agent: {e}")
        typer.echo(runt.logs(name))
        return
    typer.echo(f"\nSuccessfully created agent '{name}'")


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
    from surfkit.runtime.agent.base import AgentInstance

    active_runtimes = AgentInstance.active_runtimes()

    for runtm in active_runtimes:
        runtm.refresh()

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
            type_name = agent.type.name
            if agent.type.namespace:
                type_name = f"{agent.type.namespace}/{agent.type.name}"
            agents_list.append(
                [
                    agent.name,
                    agent.type.kind,
                    type_name,
                    agent.runtime.name(),
                    agent.status.value,
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

    runtimes = Tracker.active_runtimes()
    for runtime in runtimes:
        runtime.refresh()
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


@list_group.command("benchmarks")
def list_benchmarks():
    from taskara import Benchmark

    benchmarks = Benchmark.find()

    if not benchmarks:
        print("No benchmarks found")
    else:
        table = []
        for benchmark in benchmarks:
            table.append(
                [
                    benchmark.name,
                    benchmark.description,
                    len(benchmark.tasks),
                ]
            )

        print(
            tabulate(
                table,
                headers=[
                    "Name",
                    "Description",
                    "Num Tasks",
                ],
            )
        )
        print("")


@list_group.command("evals")
def list_evals():
    from taskara import Eval

    evals = Eval.find()

    if not evals:
        print("No evals found")
    else:
        table = []
        for eval in evals:
            table.append(
                [
                    eval.id,
                    eval.benchmark.name,
                    eval.benchmark.description,
                    len(eval.benchmark.tasks),
                ]
            )

        print(
            tabulate(
                table,
                headers=[
                    "ID",
                    "Name",
                    "Description",
                    "Num Tasks",
                ],
            )
        )
        print("")


@list_group.command("types")
def list_types():
    from typing import List

    from surfkit.config import AGENTSEA_HUB_API_URL
    from surfkit.types import AgentType

    table = []

    try:
        types = AgentType.find(remote=AGENTSEA_HUB_API_URL)
        for typ in types:
            name = typ.name
            if typ.namespace:
                name = f"{typ.namespace}/{name}"

            supports = ""
            if typ.supports:
                supports = ", ".join(typ.supports)

            tags = ""
            if typ.tags:
                tags = ", ".join(typ.tags)

            table.append(
                [name, typ.kind, typ.description, supports, tags, AGENTSEA_HUB_API_URL]
            )
    except Exception as e:
        pass

    if not table:
        types = AgentType.find()
        for typ in types:
            name = typ.name
            if typ.namespace:
                name = f"{typ.namespace}/{name}"

            supports = ""
            if typ.supports:
                supports = ", ".join(typ.supports)

            tags = ""
            if typ.tags:
                tags = ", ".join(typ.tags)

            table.append(
                [
                    name,
                    typ.kind,
                    typ.description,
                    supports,
                    tags,
                    "local",
                ]
            )

    if not table:
        print("No types found")
        return

    print(
        tabulate(
            table,
            headers=[
                "Name",
                "Kind",
                "Description",
                "Supports",
                "Tags",
                "Source",
            ],
        )
    )
    print("")


@app.command("find")
def find(help="Find an agent"):
    """Find an agent"""
    list_types()


@list_group.command("tasks")
def list_tasks(
    remote: Optional[str] = typer.Option(
        None, "--remote", "-r", help="List tasks from remote"
    ),
    tracker: Optional[str] = typer.Option(
        None, "--tracker", "-t", help="The tracker to list tasks from."
    ),
):

    from taskara import Task, V1Tasks
    from taskara.runtime.base import Tracker

    from surfkit.config import AGENTSEA_HUB_API_URL, GlobalConfig

    table = []
    if tracker:
        runtimes = Tracker.active_runtimes()
        for runtime in runtimes:
            runtime.refresh()
        trackers = Tracker.find(name=tracker)
        if not trackers:
            raise ValueError(f"Tracker '{tracker}' not found")
        trck = trackers[0]
        status, text = trck.call("/v1/tasks", "GET")
        if status != 200:
            raise ValueError(f"Failed to list tasks from tracker: {text}")
        v1tasks = V1Tasks.model_validate_json(text)
        for task in v1tasks.tasks:
            table.append(
                [
                    task.id,
                    task.description,
                    task.status,
                    trck.runtime.name(),
                ]
            )

    elif remote:
        tasks = Task.find(remote=remote)
        for task in tasks:
            table.append(
                [
                    task.id,
                    task.description,
                    task.status,
                    remote,
                ]
            )

    else:
        runtimes = Tracker.active_runtimes()
        for runtime in runtimes:
            runtime.refresh()
        trackers = Tracker.find()

        if trackers:
            for trck in trackers:
                status, text = trck.call("/v1/tasks", "GET")
                if status != 200:
                    raise ValueError(f"Failed to list tasks from tracker: {text}")
                v1tasks = V1Tasks.model_validate_json(text)
                for task in v1tasks.tasks:
                    table.append(
                        [
                            task.id,
                            task.description,
                            task.status,
                            trck.name,
                        ]
                    )

        config = GlobalConfig.read()
        if config.api_key:
            try:
                tasks = Task.find(remote=AGENTSEA_HUB_API_URL)
                for task in tasks:
                    table.append(
                        [
                            task.id,
                            task.description,
                            task.status,
                            AGENTSEA_HUB_API_URL,
                        ]
                    )
            except:
                pass

    if not table:
        typer.echo("No tasks found")
        return

    print(
        tabulate(
            table,
            headers=[
                "ID",
                "Description",
                "Status",
                "Tracker",
            ],
        )
    )
    print("")


# 'get' sub-commands
@get_group.command("agent")
def get_agent(
    name: str = typer.Argument(..., help="The name of the agent to retrieve."),
    runtime: Optional[str] = typer.Option(
        None, "--runtime", "-r", help="Get agent directly from the runtime"
    ),
):

    from surfkit.runtime.agent.base import AgentInstance

    active_runtimes = AgentInstance.active_runtimes()
    for runtm in active_runtimes:
        runtm.refresh()

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
    name: str = typer.Argument(
        ...,
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
def get_type(name: str = typer.Argument(..., help="The name of the type to retrieve.")):
    from surfkit.types import AgentType

    typer.echo(f"Getting type: {name}")
    from surfkit.config import AGENTSEA_HUB_API_URL

    types = AgentType.find(remote=AGENTSEA_HUB_API_URL, name=name)
    if not types:
        raise ValueError(f"Agent type '{name}' not found")
    agent_type = types[0]

    rich.print_json(agent_type.to_v1().model_dump_json())


@get_group.command("task")
def get_task(
    id: str = typer.Argument(..., help="ID of the task"),
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

    from surfkit.config import AGENTSEA_HUB_API_URL, GlobalConfig
    from surfkit.env import AGENTESEA_HUB_API_KEY_ENV

    config = GlobalConfig.read()
    if not config.api_key:
        raise ValueError("No API key found. Please run `surfkit login` first.")

    os.environ[AGENTESEA_HUB_API_KEY_ENV] = config.api_key

    all_tasks: List[Task] = []

    if remote:
        try:
            tasks = Task.find(remote=AGENTSEA_HUB_API_URL, id=id)
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
    name: str = typer.Argument(..., help="The name of the agent to retrieve."),
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
    name: str = typer.Argument(
        ...,
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
        typer.echo("refreshing provider...")
        _provider.refresh(log=False)
        desktop = DesktopVM.get(name)
        if not desktop:
            print(f"Desktop '{name}' not found")
            return

    if desktop:
        typer.echo(f"Deleting '{name}' desktop...")
        _provider.delete(name)
        typer.echo(f"Desktop '{name}' deleted")
    else:
        print(f"Desktop '{name}' not found")
    return


@delete_group.command("tracker")
def delete_tracker(
    name: str = typer.Argument(
        ...,
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
    name: str = typer.Argument(..., help="The name of the type."),
):
    from surfkit.config import AGENTSEA_HUB_API_URL
    from surfkit.types import AgentType

    types = AgentType.find(remote=AGENTSEA_HUB_API_URL, name=name)
    if not types:
        raise ValueError(f"Agent type '{type}' not found")
    agent_type = types[0]
    agent_type.remove()
    typer.echo(f"Agent type '{name}' deleted")


# View subcommands


@view_group.command("device")
def view_device(
    name: str = typer.Argument(..., help="The name of the device to view."),
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
    from surfkit.config import AGENTSEA_HUB_URL, GlobalConfig

    url = urljoin(AGENTSEA_HUB_URL, "cli-login")
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
    build: bool = typer.Option(False, "--build", "-b", help="Build the docker image"),
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

    from surfkit.config import AGENTSEA_HUB_API_URL, GlobalConfig

    url = urljoin(AGENTSEA_HUB_API_URL, "v1/agenttypes")
    typer.echo(f"\nPublishing agent to {url}...\n")

    config = GlobalConfig.read()
    if not config.api_key:
        raise ValueError("No API key found. Please run `surfkit login` first.")

    headers = {"Authorization": f"Bearer {config.api_key}"}
    resp = requests.post(url, json=typ.to_v1().model_dump(), headers=headers)
    resp.raise_for_status()
    typer.echo(f"Agent '{typ.name}' published")


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


@app.command(help="Build an agent container")
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
    description: str = typer.Argument(..., help="Description of the task."),
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
        False, "--kill", "-k", help="Whether to kill the agent when done."
    ),
    view: bool = typer.Option(True, "--view", "-v", help="Whether to view the device."),
    follow: bool = typer.Option(True, help="Whether to follow the agent logs."),
    starting_url: Optional[str] = typer.Option(
        None, help="Starting URL if applicable."
    ),
    auth_enabled: bool = typer.Option(
        False,
        "--auth-enabled",
        "-e",
        help="Whether to enable auth for the agent.",
    ),
    local_keys: bool = typer.Option(
        False, "--local-keys", "-l", help="Use local API keys."
    ),
    debug: bool = typer.Option(False, help="Run the agent with debug logging"),
):
    from surfkit.client import solve

    solve(
        description=description,
        agent=agent,
        agent_runtime=agent_runtime,
        agent_type=agent_type,
        agent_file=agent_file,
        agent_version=agent_version,
        device=device,
        device_type=device_type,
        device_provider=device_provider,
        tracker=tracker,
        tracker_runtime=tracker_runtime,
        tracker_remote=tracker_remote,
        max_steps=max_steps,
        kill=kill,
        view=view,
        follow=follow,
        starting_url=starting_url,
        auth_enabled=auth_enabled,
        local_keys=local_keys,
        debug=debug,
        interactive=True,
        create_tracker=False,
    )


# @app.command("eval", help="Evaluate an agent on a benchmark")  # TODO
def eval(
    benchmark: str = typer.Argument(
        help="Benchmark name",
    ),
    parallel: int = typer.Option(
        4,
        "--parallel",
        "-p",
        help="Max number of parallel runs",
    ),
    agent_type: str = typer.Option(
        None,
        "--agent-type",
        "-a",
        help="The agent type to use for the benchmark",
    ),
    agent: str = typer.Option(
        None,
        "--agent",
        "-n",
        help="The agent to use for the benchmark",
    ),
    agent_file: str = typer.Option(
        None,
        "--agent-file",
        "-f",
        help="The agent file to use for the benchmark",
    ),
    tracker: str = typer.Option(
        None,
        "--tracker",
        "-t",
        help="The tracker to use for the benchmark",
    ),
):
    import yaml
    from taskara import V1Benchmark
    from taskara.runtime.base import Tracker

    if not tracker:
        trackers = Tracker.find()
        if not trackers:
            typer.echo("No trackers found")
            raise typer.Exit()
        trck = trackers[0]
    else:
        trackers = Tracker.find(name=tracker)
        if not trackers:
            typer.echo(f"Tracker '{tracker}' not found")
            raise typer.Exit()
        trck = trackers[0]

    typer.echo("you wish...")

    # with open(file) as f:
    #     dct = yaml.safe_load(f)
    #     v1_benchmark = V1Benchmark.model_validate(dct)

    # status, text = trck.call(
    #     "/v1/benchmarks", method="POST", data=v1_benchmark.model_dump()
    # )
    # if status != 200:
    #     typer.echo(f"Error creating benchmark: {text}")
    #     raise typer.Exit()

    # typer.echo(f"Benchmark '{v1_benchmark.name}' created using '{trck.name}' tracker")


@logs_group.command("agent")
def get_agent_logs(
    name: str = typer.Argument(
        ...,
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
    from surfkit.runtime.agent.base import AgentInstance

    active_runtimes = AgentInstance.active_runtimes()
    for runtm in active_runtimes:
        runtm.refresh()

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


@logs_group.command("tracker")
def get_tracker_logs(
    name: str = typer.Argument(
        ...,
        help="The name of the tracker whose logs are to be retrieved.",
    ),
    follow: bool = typer.Option(
        False, "--follow", "-f", help="Whether to continuously follow the logs."
    ),
):
    """
    Retrieve tracker logs
    """
    from taskara.runtime.base import Tracker

    active_runtimtes = Tracker.active_runtimes()
    for runtime in active_runtimtes:
        runtime.refresh()

    trackers = Tracker.find(name=name)
    if not trackers:
        typer.echo(f"Tracker '{name}' not found")
        raise typer.Exit(1)

    tracker = trackers[0]
    logs = tracker.logs(follow=follow)
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


@app.command("config")
def config():
    """
    Retrieve tracker logs
    """
    from surfkit.config import (
        AGENTSEA_AUTH_URL,
        AGENTSEA_HUB_API_URL,
        AGENTSEA_HUB_URL,
        GlobalConfig,
    )

    typer.echo(f"Hub URL: {AGENTSEA_HUB_URL}")
    typer.echo(f"Hub API URL: {AGENTSEA_HUB_API_URL}")
    typer.echo(f"Auth URL: {AGENTSEA_AUTH_URL}")

    config = GlobalConfig.read()
    if config.api_key:
        typer.echo(f"API key: {config.api_key}")


if __name__ == "__main__":
    app()
