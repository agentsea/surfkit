import logging
import os
from typing import Iterator, List, Optional, Union

import requests
import yaml
from agentdesk import Desktop
from agentdesk.vm import DesktopVM
from agentdesk.vm.load import load_provider
from devicebay import V1DeviceType
from namesgenerator import get_random_name
from taskara import Benchmark, Eval, Task, V1Benchmark, V1Tasks
from taskara.runtime.base import Tracker
from taskara.task import Task

from surfkit.config import AGENTSEA_HUB_API_URL, GlobalConfig
from surfkit.env_opts import find_envs
from surfkit.runtime.agent.base import AgentInstance
from surfkit.runtime.agent.docker import DockerAgentRuntime, DockerConnectConfig
from surfkit.runtime.agent.kube import KubeAgentRuntime, KubeConnectConfig
from surfkit.runtime.agent.process import ProcessAgentRuntime, ProcessConnectConfig
from surfkit.server.models import V1AgentType
from surfkit.types import AgentType
from surfkit.util import find_open_port

logger = logging.getLogger(__name__)


class Client:

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.headers = {"Authorization": f"Bearer {self.api_key}"} if api_key else {}

    def create_device(
        self,
        name: Optional[str] = None,
        type: Optional[str] = "desktop",
        provider: str = "qemu",
        image: Optional[str] = None,
        memory: int = 4,
        cpu: int = 2,
        disk: str = "30gb",
        reserve_ip: bool = False,
    ):
        """Create a new device"""
        from agentdesk.server.models import V1ProviderData
        from agentdesk.vm.load import load_provider

        if type != "desktop":
            logger.error("Currently only 'desktop' type is supported.")
            raise ValueError("Currently only 'desktop' type is supported.")

        if not name:
            name = get_random_name(sep="-")

        if provider == "ec2":
            data = V1ProviderData(type=provider, args={"region": "us-east-1"})
            _provider = load_provider(data)
        else:
            data = V1ProviderData(type=provider)
            _provider = load_provider(data)

        logger.info(f"Creating desktop '{name}' using '{provider}' provider")
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
            logger.error("Keyboard interrupt received, exiting...")
            raise
        except Exception as e:
            logger.error(f"Error occurred: {e}")
            raise

    def create_tracker(
        self,
        name: Optional[str] = None,
        runtime: str = "docker",
        image: str = "us-central1-docker.pkg.dev/agentsea-dev/taskara/api:latest",
        auth_enabled: bool = False,
    ):
        """Create a new tracker"""

        # Generate a random name if not provided
        if not name:
            name = get_random_name(sep="-")
            if not name:
                raise SystemError("Name is required for tracker")

        if runtime == "docker":
            from taskara.runtime.docker import DockerConnectConfig, DockerTrackerRuntime

            runt = DockerTrackerRuntime(DockerConnectConfig(image=image))

        elif runtime == "kube":
            from taskara.runtime.kube import KubeConnectConfig, KubeTrackerRuntime

            runt = KubeTrackerRuntime(KubeConnectConfig(image=image))

        else:
            logger.error(f"Invalid runtime: {runtime}")
            raise ValueError(f"Invalid runtime: {runtime}")

        server = runt.run(name=name, auth_enabled=auth_enabled)
        logger.info(f"Tracker '{name}' created using '{runtime}' runtime")
        return server

    def create_benchmark(self, file: str, tracker: Optional[str] = None):
        """Create a new benchmark"""

        if not tracker:
            trackers = Tracker.find()
            if not trackers:
                logger.error("No trackers found")
                raise ValueError("No trackers found")
            trck = trackers[0]
        else:
            trackers = Tracker.find(name=tracker)
            if not trackers:
                logger.error(f"Tracker '{tracker}' not found")
                raise ValueError(f"Tracker '{tracker}' not found")
            trck = trackers[0]

        with open(file) as f:
            dct = yaml.safe_load(f)
            v1_benchmark = V1Benchmark.model_validate(dct)

        status, text = trck.call(
            "/v1/benchmarks", method="POST", data=v1_benchmark.model_dump()
        )
        if status != 200:
            logger.error(f"Error creating benchmark: {text}")
            raise ValueError(f"Error creating benchmark: {text}")

        logger.info(
            f"Benchmark '{v1_benchmark.name}' created using '{trck.name}' tracker"
        )
        return v1_benchmark

    def create_task(
        self,
        description: str,
        assigned_to: Optional[str] = None,
        assigned_type: Optional[str] = None,
        device: Optional[str] = None,
        device_type: Optional[str] = None,
        max_steps: int = 30,
        tracker: Optional[str] = None,
    ):
        """Create a new task"""
        _device_type = None
        if device_type:
            _device_type = V1DeviceType(name=device_type)

        _device = None
        if device:
            desk_vms = Desktop.find(name=device)
            if not desk_vms:
                logger.error(f"Desktop '{device}' not found")
                raise ValueError(f"Desktop '{device}' not found")
            _desk_vm = desk_vms[0]

            _desk = Desktop.from_vm(_desk_vm)
            _device = _desk.to_v1()

        if tracker:
            trackers = Tracker.find(name=tracker)
            if not trackers:
                logger.error(f"Tracker '{tracker}' not found")
                raise ValueError(f"Tracker '{tracker}' not found")
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

        logger.info(f"Task '{task.id}' created")
        return task

    def create_agent(
        self,
        runtime: str = "process",
        file: str = "./agent.yaml",
        name: Optional[str] = None,
        type: Optional[str] = None,
        auth_enabled: bool = False,
        local_keys: bool = False,
        debug: bool = False,
    ):
        """Create a new agent"""

        if not runtime:
            if type:
                runtime = "docker"
            else:
                runtime = "process"

        if runtime == "docker":
            from surfkit.runtime.agent.docker import (
                DockerAgentRuntime,
                DockerConnectConfig,
            )

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
        logger.info(
            f"Running agent '{type or file}' with runtime '{runtime}' and name '{name}'..."
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
            logger.error(f"Failed to run agent: {e}")
            logger.error(runt.logs(name))
            raise

        logger.info(f"Successfully created agent '{name}'")
        return name

    def list_agents(self, runtime: Optional[str] = None) -> List[AgentInstance]:
        agents_list = []

        active_runtimes = AgentInstance.active_runtimes()
        for runtm in active_runtimes:
            runtm.refresh()

        if runtime:
            if runtime == "docker" or runtime == "all":
                try:
                    dconf = DockerConnectConfig()
                    runt = DockerAgentRuntime.connect(cfg=dconf)
                    agents_list.extend(runt.list())
                except Exception as e:
                    if runtime != "all":
                        raise
                    logger.error(f"Failed to list agents for Docker runtime: {e}")

            if runtime == "kube" or runtime == "all":
                try:
                    kconf = KubeConnectConfig()
                    runt = KubeAgentRuntime.connect(cfg=kconf)
                    agents_list.extend(runt.list())
                except Exception as e:
                    if runtime != "all":
                        raise
                    logger.error(f"Failed to list agents for Kubernetes runtime: {e}")

            if runtime == "process" or runtime == "all":
                try:
                    pconf = ProcessConnectConfig()
                    runt = ProcessAgentRuntime.connect(cfg=pconf)
                    agents_list.extend(runt.list())
                except Exception as e:
                    if runtime != "all":
                        raise
                    logger.error(f"Failed to list agents for Process runtime: {e}")

        else:
            agents_list = AgentInstance.find()

        return agents_list

    def list_devices(self, provider: Optional[str] = None) -> List[DesktopVM]:
        provider_is_refreshed = {}
        devices_list = []

        vms = DesktopVM.find()
        if not vms:
            return devices_list
        else:
            for desktop in vms:
                if not desktop.provider:
                    continue
                if provider and desktop.provider.type != provider:
                    continue

                _provider = load_provider(desktop.provider)
                if not provider_is_refreshed.get(desktop.provider.type):
                    if not desktop.reserved_ip:
                        _provider.refresh(log=False)
                        provider_is_refreshed[desktop.provider.type] = True
                        desktop = DesktopVM.get(desktop.name)
                        if not desktop:
                            continue

                devices_list.append(desktop)

        return devices_list

    def list_trackers(self) -> List[Tracker]:
        trackers_list = []

        runtimes = Tracker.active_runtimes()
        for runtime in runtimes:
            runtime.refresh()
        trackers_list = Tracker.find()

        return trackers_list

    def list_benchmarks(self) -> List[Benchmark]:
        benchmarks_list = Benchmark.find()
        return benchmarks_list

    def list_evals(self) -> List[Eval]:
        evals_list = Eval.find()
        return evals_list

    def list_types(self) -> List[AgentType]:
        all_types: List[AgentType] = []

        try:
            types = AgentType.find(remote=AGENTSEA_HUB_API_URL)
            all_types.extend(types)
        except Exception as e:
            pass

        if not all_types:
            types = AgentType.find()
            all_types.extend(types)

        return all_types

    def list_tasks(
        self, remote: Optional[str] = None, tracker: Optional[str] = None
    ) -> List[Task]:
        tasks_list = []

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
            tasks_list.extend(v1tasks.tasks)

        elif remote:
            tasks_list = Task.find(remote=remote)

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
                    tasks_list.extend(v1tasks.tasks)

            config = GlobalConfig.read()
            if config.api_key:
                try:
                    tasks_list = Task.find(remote=AGENTSEA_HUB_API_URL)
                except:
                    pass

        return tasks_list

    def get_agent(self, name: str, runtime: Optional[str] = None) -> AgentInstance:
        active_runtimes = AgentInstance.active_runtimes()
        for runtm in active_runtimes:
            runtm.refresh()

        if runtime:
            if runtime == "docker":
                dconf = DockerConnectConfig()
                runt = DockerAgentRuntime.connect(cfg=dconf)
            elif runtime == "kube":
                kconf = KubeConnectConfig()
                runt = KubeAgentRuntime.connect(cfg=kconf)
            elif runtime == "process":
                pconf = ProcessConnectConfig()
                runt = ProcessAgentRuntime.connect(cfg=pconf)
            else:
                raise ValueError(f"Unknown runtime '{runtime}'")
            instance = runt.get(name)
        else:
            instances = AgentInstance.find(name=name)
            if not instances:
                raise ValueError(f"Agent instance '{name}' not found")
            instance = instances[0]

        return instance

    def get_device(self, name: str, provider: Optional[str] = None) -> DesktopVM:
        desktop = DesktopVM.get(name)
        if not desktop:
            raise ValueError("Desktop not found")
        if not desktop.provider:
            raise ValueError("No desktop provider")
        if provider and desktop.provider.type != provider:
            raise ValueError(f"Desktop '{name}' not found")

        _provider = load_provider(desktop.provider)
        if not desktop.reserved_ip:
            _provider.refresh(log=False)
            desktop = DesktopVM.get(name)
            if not desktop:
                raise ValueError(f"Desktop '{name}' not found")

        return desktop

    def get_type(self, name: str) -> AgentType:
        types = AgentType.find(remote=AGENTSEA_HUB_API_URL, name=name)
        if not types:
            raise ValueError(f"Agent type '{name}' not found")
        agent_type = types[0]

        return agent_type

    def get_task(self, id: str, remote: Optional[str] = None) -> Task:
        config = GlobalConfig.read()
        if not config.api_key:
            raise ValueError("No API key found. Please run `surfkit login` first.")

        from surfkit.env import AGENTESEA_HUB_API_KEY_ENV

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

        return task

    def delete_agent(
        self, name: str, runtime: Optional[str] = None, force: bool = False
    ) -> None:
        if runtime:
            if runtime == "docker":
                dconf = DockerConnectConfig()
                runt = DockerAgentRuntime.connect(cfg=dconf)
            elif runtime == "kube":
                kconf = KubeConnectConfig()
                runt = KubeAgentRuntime.connect(cfg=kconf)
            elif runtime == "process":
                pconf = ProcessConnectConfig()
                runt = ProcessAgentRuntime.connect(cfg=pconf)
            else:
                raise ValueError(f"Unknown runtime '{runtime}'")

            runt.delete(name)
            logger.info(f"Agent '{name}' deleted")

        else:
            agents = AgentInstance.find(name=name)
            if not agents:
                raise ValueError(f"Agent '{name}' not found")
            agent = agents[0]
            agent.delete(force=force)
            logger.info(f"Agent '{name}' deleted")

    def delete_device(self, name: str, provider: Optional[str] = None) -> None:
        desktop = DesktopVM.get(name)
        if not desktop:
            raise ValueError("Desktop not found")
        if not desktop.provider:
            raise ValueError("No desktop provider")
        if provider and desktop.provider.type != provider:
            raise ValueError(f"Desktop '{name}' not found")

        _provider = load_provider(desktop.provider)
        if not desktop.reserved_ip:
            logger.info("Refreshing provider...")
            _provider.refresh(log=False)
            desktop = DesktopVM.get(name)
            if not desktop:
                raise ValueError(f"Desktop '{name}' not found")

        logger.info(f"Deleting '{name}' desktop...")
        _provider.delete(name)
        logger.info(f"Desktop '{name}' deleted")

    def delete_tracker(self, name: str, force: bool = False) -> None:
        trackers = Tracker.find(name=name)
        if not trackers:
            raise ValueError(f"Tracker '{name}' not found")

        tracker = trackers[0]
        tracker.delete(force=force)
        logger.info("Tracker deleted")

    def delete_type(self, name: str) -> None:
        types = AgentType.find(remote=AGENTSEA_HUB_API_URL, name=name)
        if not types:
            raise ValueError(f"Agent type '{name}' not found")
        agent_type = types[0]
        agent_type.remove()
        logger.info(f"Agent type '{name}' deleted")

    def view_device(self, name: str, background: bool):
        """View a device"""
        # Implement device viewing logic
        from agentdesk.vm import DesktopVM

        desktop = DesktopVM.get(name)
        if not desktop:
            raise ValueError(f"Desktop '{name}' not found")

        desktop.view(background=background)

    def solve(
        self,
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
        view: bool = True,
        follow: bool = True,
        starting_url: Optional[str] = None,
        auth_enabled: bool = False,
        local_keys: bool = False,
        debug: bool = False,
    ) -> Task:
        from agentdesk import Desktop
        from taskara import Task
        from taskara.runtime.base import Tracker

        from surfkit.cli.util import tracker_addr_agent, tracker_addr_local
        from surfkit.config import AGENTSEA_HUB_API_URL, GlobalConfig
        from surfkit.server.models import V1SolveTask
        from surfkit.types import AgentType

        if not agent_runtime:
            if agent_file:
                agent_runtime = "process"
            elif agent_type:
                agent_runtime = "docker"
            else:
                agent_runtime = "docker"

        _task_token = None
        if tracker:

            trackers = Tracker.find(name=tracker)
            if not trackers:
                raise ValueError(f"Expected tracker with name '{tracker}'")
            _tracker = trackers[0]
            _tracker_agent_addr = tracker_addr_agent(_tracker, agent_runtime)
            _tracker_local_addr = tracker_addr_local(_tracker)

        elif tracker_runtime:

            if tracker_runtime == "docker":
                from taskara.runtime.docker import DockerTrackerRuntime

                task_runt = DockerTrackerRuntime()

            elif tracker_runtime == "kube":
                from taskara.runtime.kube import KubeTrackerRuntime

                task_runt = KubeTrackerRuntime()

            else:
                raise ValueError(f"Invalid runtime: {tracker_runtime}")

            name = get_random_name(sep="-")
            if not name:
                raise SystemError("Name is required for tracker")

            _tracker = task_runt.run(name=name, auth_enabled=auth_enabled)
            _tracker_agent_addr = tracker_addr_agent(_tracker, agent_runtime)
            _tracker_local_addr = tracker_addr_local(_tracker)

        elif tracker_remote:
            _tracker_agent_addr = tracker_remote
            _tracker_local_addr = tracker_remote

        else:
            config = GlobalConfig.read()
            if config.api_key:
                _tracker_agent_addr = AGENTSEA_HUB_API_URL
                _tracker_local_addr = AGENTSEA_HUB_API_URL
                _task_token = config.api_key
            else:
                trackers = Tracker.find(runtime_name=agent_runtime)
                if not trackers:
                    raise ValueError("No trackers found or specified")
                task_svr = trackers[0]
                _tracker_agent_addr = tracker_addr_agent(task_svr, agent_runtime)
                _tracker_local_addr = tracker_addr_local(task_svr)

        runt = None
        if agent:
            active_runtimes = AgentInstance.active_runtimes()
            for runtm in active_runtimes:
                runtm.refresh()

            instances = AgentInstance.find(name=agent)
            if not instances:
                raise ValueError(f"Expected instances of '{agent}'")
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
                from surfkit.runtime.agent.kube import (
                    KubeAgentRuntime,
                    KubeConnectConfig,
                )

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
        vm = None
        if device_type:
            if device_type == "desktop":
                from agentdesk.server.models import V1ProviderData
                from agentdesk.vm.load import load_provider

                if device_provider == "qemu" and agent_runtime != "process":
                    raise ValueError(
                        f"QEMU provider is only supported for the agent 'process' runtime"
                    )

                data = V1ProviderData(type=device_provider)
                _provider = load_provider(data)

                try:
                    vm = _provider.create(name=agent)
                    _device = Desktop.from_vm(vm)
                    v1device = _device.to_v1()
                except KeyboardInterrupt:
                    raise

            else:
                raise ValueError(f"unknown device type {device_type}")

        if device:
            vms = Desktop.find(name=device)
            if not vms:
                raise ValueError(f"Device '{device}' not found")
            vm = vms[0]

            if (
                vm.provider
                and vm.provider.type == "qemu"
                and agent_runtime != "process"
            ):
                raise ValueError(
                    "Qemu desktop can only be used with the agent 'process' runtime"
                )

            _device = Desktop.from_vm(vm)
            v1device = _device.to_v1()

        if agent_type:
            all_types: List[AgentType] = []
            type_parts = agent_type.split("/")

            if len(type_parts) == 1:
                types = AgentType.find(name=agent_type)
                if types:
                    all_types.extend(types)
            elif len(type_parts) == 2:
                config = GlobalConfig.read()
                if config.api_key:
                    types = AgentType.find(
                        remote=AGENTSEA_HUB_API_URL,
                        namespace=type_parts[0],
                        name=type_parts[1],
                    )
                    if types:
                        all_types.extend(types)

            if not all_types:
                raise ValueError(f"Agent type '{agent_type}' not found")

            typ = all_types[0]
            if not agent:
                agent = get_random_name("-")
                if not agent:
                    raise ValueError("could not generate agent name")

            from surfkit.env_opts import find_envs

            env_vars = find_envs(typ, local_keys)
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
                logger.error(runt.logs(agent))
                raise
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
                logger.error(runt.logs(agent))
                raise
            agent = instance.name

        if not agent:
            raise ValueError("Either agent or agent_type needs to be provided")

        params = {}
        if starting_url:
            params["site"] = starting_url

        owner = "tom@myspace.com"
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
        )

        if _device and view:
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

        task._remote = _tracker_agent_addr
        solve_v1 = V1SolveTask(task=task.to_v1())
        runt.solve_task(agent, solve_v1, follow_logs=follow, attach=kill)

        if kill and not follow:
            try:
                runt.delete(agent)
                instances = AgentInstance.find(name=agent)
                if instances:
                    instances[0].delete(force=True)
            except:
                pass

        return task

    def evaluate_agent(
        self,
        benchmark: str,
        parallel: int,
        agent_type: str,
        agent: str,
        agent_file: str,
        tracker: str,
    ):
        """Evaluate an agent on a benchmark"""
        # Implement agent evaluation logic
        raise NotImplementedError()

    def get_agent_logs(
        self, name: str, runtime: Optional[str] = None, follow: bool = False
    ) -> Union[str, Iterator[str]]:
        """
        Retrieve agent logs
        """
        active_runtimes = AgentInstance.active_runtimes()
        for runtm in active_runtimes:
            runtm.refresh()

        if runtime:
            if runtime == "docker":
                config = DockerConnectConfig()
                runtime_instance = DockerAgentRuntime.connect(cfg=config)
            elif runtime == "kube":
                config = KubeConnectConfig()
                runtime_instance = KubeAgentRuntime.connect(cfg=config)
            elif runtime == "process":
                config = ProcessConnectConfig()
                runtime_instance = ProcessAgentRuntime.connect(cfg=config)
            else:
                raise ValueError(f"Unsupported runtime: {runtime}")

            # Fetch logs using the AgentRuntime instance
            try:
                logs = runtime_instance.logs(name, follow)
                return logs
            except Exception as e:
                raise RuntimeError(f"Failed to retrieve logs: {str(e)}")
        else:
            instances = AgentInstance.find(name=name)
            if not instances:
                raise ValueError(f"Agent '{name}' not found")
            instance = instances[0]
            logs = instance.logs(follow=follow)
            return logs

    def get_tracker_logs(
        self, name: str, follow: bool = False
    ) -> Union[str, Iterator[str]]:
        """
        Retrieve tracker logs
        """
        active_runtimes = Tracker.active_runtimes()
        for runtime in active_runtimes:
            runtime.refresh()

        trackers = Tracker.find(name=name)
        if not trackers:
            raise ValueError(f"Tracker '{name}' not found")

        tracker = trackers[0]
        logs = tracker.logs(follow=follow)
        return logs
