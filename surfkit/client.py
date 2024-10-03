import logging
from typing import Optional, Dict

from agentdesk.device_v1 import Desktop
from namesgenerator import get_random_name
from taskara import Task
from taskara.runtime.base import Tracker
import typer

from surfkit.config import AGENTSEA_HUB_API_URL, GlobalConfig
from surfkit.server.models import V1SolveTask
from surfkit.types import AgentType
from surfkit.runtime.agent.base import AgentInstance

logger = logging.getLogger(__name__)


def solve(
    description: str,
    agent: Optional[str] = None,
    agent_runtime: Optional[str] = None,
    agent_type: Optional[str] = None,
    agent_file: Optional[str] = None,
    agent_version: Optional[str] = None,
    device: Optional[str] = None,
    device_type: Optional[str] = None,
    device_provider: Optional[str] = None,
    tracker: Optional[str] = None,
    tracker_runtime: Optional[str] = None,
    tracker_remote: Optional[str] = None,
    max_steps: int = 30,
    kill: bool = False,
    view: bool = False,
    follow: bool = True,
    starting_url: Optional[str] = None,
    auth_enabled: bool = False,
    local_keys: bool = False,
    parent_id: Optional[str] = None,
    debug: bool = False,
    create_tracker: bool = True,
    interactive: bool = False,
) -> Task:
    """Solve a task using an agent.

    Args:
        description (str): Description of the task.
        agent (Optional[str], optional): Agent to use. Defaults to None.
        agent_runtime (Optional[str], optional): Agent runtime to use. Defaults to None.
        agent_type (Optional[str], optional): Create a new agent using an agent type. Defaults to None.
        agent_file (Optional[str], optional): Create a new agent using an agent file. Defaults to None.
        agent_version (Optional[str], optional): Version of the agent. Defaults to None.
        device (Optional[str], optional): Device to run the task on. Defaults to None.
        device_type (Optional[str], optional): Create a device using the device type, then run the task. Defaults to None.
        device_provider (Optional[str], optional): Provider of the device if creating one. Defaults to None.
        tracker (Optional[str], optional): Tracker to use. Defaults to None.
        tracker_runtime (Optional[str], optional): Tracker runtime to use if creating one. Defaults to None.
        tracker_remote (Optional[str], optional): Address of a remote tracker to use. Defaults to None.
        max_steps (int, optional): Max steps allowed to solve the task. Defaults to 30.
        kill (bool, optional): Whether to kill the agent at the end. Defaults to False.
        view (bool, optional): Whether to launch a browser and view the task. Defaults to False.
        follow (bool, optional): Whether to follow the logs. Defaults to True.
        starting_url (Optional[str], optional): Starting URL, only applies to desktops and browsers. Defaults to None.
        auth_enabled (bool, optional): Whether auth should be enabled on the agent. Defaults to False.
        local_keys (bool, optional): Whether to use local LLM provider keys the agent requires. Defaults to False.
        parent_id (Optional[str], optional): Parent ID of the task. Defaults to None.
        debug (bool, optional): Whether to turn on debug logging. Defaults to False.
        create_tracker (bool, optional): Whether to create the tracker if not present. Defaults to True.
        interactive (bool, optional): Whether to run in interactive mode. Defaults to False.

    Returns:
        Task: The task
    """
    if not agent_runtime:
        if agent_file:
            agent_runtime = "process"
        elif agent_type:
            agent_runtime = "docker"
        else:
            agent_runtime = "docker"

    runt = None
    if agent:
        active_runtimes = AgentInstance.active_runtimes()
        for runtm in active_runtimes:
            runtm.refresh()

        instances = AgentInstance.find(name=agent)
        if not instances:
            raise ValueError(f"Expected instances of '{agent}'")
        logger.info(f"Found agent instance '{agent}'")
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

    agent_runtime = runt.name()

    _task_token = None
    if tracker:
        from surfkit.cli.util import tracker_addr_agent, tracker_addr_local

        trackers = Tracker.find(name=tracker)
        if not trackers:
            raise ValueError(f"Expected tracker with name '{tracker}'")
        _tracker = trackers[0]
        _tracker_agent_addr = tracker_addr_agent(_tracker, agent_runtime)
        _tracker_local_addr = tracker_addr_local(_tracker)

    elif tracker_runtime:
        from surfkit.cli.util import tracker_addr_agent, tracker_addr_local

        if tracker_runtime == "docker":
            from taskara.runtime.docker import DockerTrackerRuntime

            task_runt = DockerTrackerRuntime()

        elif tracker_runtime == "kube":
            from taskara.runtime.kube import KubeTrackerRuntime

            task_runt = KubeTrackerRuntime()

        else:
            logger.error(f"Invalid runtime: {tracker_runtime}")
            raise ValueError(f"Invalid runtime: {tracker_runtime}")

        name = get_random_name(sep="-")
        if not name:
            raise SystemError("Name is required for tracker")

        _tracker = task_runt.run(name=name, auth_enabled=auth_enabled)
        logger.info(f"Tracker '{name}' created using '{tracker_runtime}' runtime")

        _tracker_agent_addr = tracker_addr_agent(_tracker, agent_runtime)
        _tracker_local_addr = tracker_addr_local(_tracker)

    elif tracker_remote:
        _tracker_agent_addr = tracker_remote
        _tracker_local_addr = tracker_remote

    else:
        from surfkit.cli.util import tracker_addr_agent, tracker_addr_local

        config = GlobalConfig.read()
        if config.api_key:
            _tracker_agent_addr = AGENTSEA_HUB_API_URL
            _tracker_local_addr = AGENTSEA_HUB_API_URL
            _task_token = config.api_key
        else:
            trackers = Tracker.find(runtime_name=agent_runtime)
            if not trackers:
                if interactive and not create_tracker:
                    create = typer.confirm(
                        "No trackers found. Would you like to create one?"
                    )
                    create_tracker = create
                if create_tracker:
                    from taskara.runtime.docker import DockerTrackerRuntime
                    from taskara.runtime.kube import KubeTrackerRuntime

                    if agent_runtime == "docker" or agent_runtime == "process":
                        task_runt = DockerTrackerRuntime()
                    elif agent_runtime == "kube":
                        task_runt = KubeTrackerRuntime()
                    else:
                        logger.error(f"Invalid runtime: {agent_runtime}")
                        raise ValueError(f"Invalid runtime: {agent_runtime}")

                    name = get_random_name(sep="-")
                    if not name:
                        raise SystemError("Name is required for tracker")

                    task_svr = task_runt.run(name=name, auth_enabled=auth_enabled)
                    logger.info(
                        f"Tracker '{name}' created using '{task_runt.name()}' runtime"
                    )
                else:
                    raise ValueError(
                        "`tracker`, `tracker_runtime`, or `tracker_remote` flag must be provided. Or a tracker must be running, or a hub API key must be present"
                    )
            else:
                task_svr = trackers[0]
                logger.info(
                    f"Using tracker '{task_svr.name}' running on '{task_svr.runtime.name()}'"
                )

            _tracker_agent_addr = tracker_addr_agent(task_svr, agent_runtime)
            _tracker_local_addr = tracker_addr_local(task_svr)

    v1device = None
    _device = None
    if device_type:
        if device_type == "desktop":
            from agentdesk.server.models import V1ProviderData
            from agentdesk.runtime.load import load_provider

            if device_provider == "qemu" and agent_runtime != "process":
                raise ValueError(
                    f"QEMU provider is only supported for the agent 'process' runtime"
                )

            data = V1ProviderData(type=device_provider)
            _provider = load_provider(data)

            logger.info(
                f"Creating desktop '{agent}' using '{device_provider}' provider"
            )
            try:
                vm = _provider.create(
                    name=agent,
                )
                _device = Desktop.from_instance(vm)
                v1device = _device.to_v1()
            except KeyboardInterrupt:
                print("Keyboard interrupt received, exiting...")
                raise
        else:
            raise ValueError(f"unknown device type {device_type}")

    vm = None
    if device:
        logger.info(f"finding device '{device}'...")
        vms = Desktop.find(name=device)
        if not vms:
            raise ValueError(f"Device '{device}' not found")
        vm = vms[0]

        if vm.provider and vm.provider.type == "qemu":
            if agent_runtime != "process" and agent_runtime != "docker":
                raise ValueError(
                    "Qemu desktop can only be used with the agent 'process' or 'docker' runtime"
                )
        _device = Desktop.from_instance(vm)
        v1device = _device.to_v1()
        logger.info(f"found device '{device}'...")

    if agent_type:
        from typing import List

        all_types: List[AgentType] = []

        type_parts = agent_type.split("/")

        if len(type_parts) == 1:
            types = AgentType.find(name=agent_type)
            if types:
                all_types.extend(types)
        elif len(type_parts) == 2:
            try:
                config = GlobalConfig.read()
                if config.api_key:
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
            raise ValueError(f"Agent type '{agent_type}' not found")

        typ = all_types[0]
        if not agent:
            agent = get_random_name("-")
            if not agent:
                raise ValueError("could not generate agent name")

        from surfkit.env_opts import find_envs

        env_vars = find_envs(typ, local_keys)

        logger.info(f"creating agent {agent}...")
        try:
            instance = runt.run(
                agent_type=typ,
                name=agent,
                version=agent_version,
                auth_enabled=auth_enabled,
                env_vars=env_vars,
                debug=debug,
            )
        except Exception as e:
            logger.error(f"Failed to run agent: {e}")
            logger.info(runt.logs(agent))
            raise ValueError(f"Failed to run agent: {e}")
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

        from surfkit.env_opts import find_envs

        env_vars = find_envs(typ, local_keys)

        logger.info(f"creating agent {agent} from file {agent_file}...")
        try:
            instance = runt.run(
                agent_type=typ,
                name=agent,
                version=agent_version,
                auth_enabled=auth_enabled,
                env_vars=env_vars,
                debug=debug,
            )
        except Exception as e:
            logger.error(f"Failed to run agent: {e}")
            logger.info(runt.logs(agent))
            raise ValueError(f"Failed to run agent: {e}")
        agent = instance.name

    if not agent:
        raise ValueError("Either agent or agent_type needs to be provided")

    params = {}
    if starting_url:
        params["site"] = starting_url

    owner = "anonymous@agentsea.ai"
    config = GlobalConfig.read()
    if config.api_key:
        from surfkit.hub import HubAuth

        hub = HubAuth()
        user_info = hub.get_user_info(config.api_key)
        owner = user_info.email

    task = Task(
        description=description,
        parameters=params,
        max_steps=max_steps,
        device=v1device,
        assigned_to=agent,
        assigned_type=agent_type,
        remote=_tracker_local_addr,
        owner_id=owner,
        auth_token=_task_token,
        parent_id=parent_id,
    )

    if _device and view:
        logger.info("viewing device...")
        from surfkit.cli.view import view as _view

        instances = AgentInstance.find(name=agent)
        if not instances:
            raise ValueError(f"agent '{agent}' not found")
        instance = instances[0]

        if not vm:
            raise ValueError("vm not found for ui")

        _view(
            desk=vm,
            agent=instance,
            tracker_addr=_tracker_local_addr,
            background=True,
            task_id=task.id,
            auth_token=_task_token,
        )

    logger.info(f"Solving task '{task.description}' with agent '{agent}'...")
    task._remote = _tracker_agent_addr
    solve_v1 = V1SolveTask(task=task.to_v1())
    runt.solve_task(agent, solve_v1, follow_logs=follow, attach=kill)

    if kill and not follow:
        logger.info(f"Killing agent {agent}...")
        try:
            runt.delete(agent)
            instances = AgentInstance.find(name=agent)
            if instances:
                instances[0].delete(force=True)
        except:
            pass
    return task


def learn(
    description: str,
    parameters: Dict[str, str] = {},
    device_type: str = "desktop",
    starting_url: Optional[str] = None,
):
    pass
