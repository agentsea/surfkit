import json
import logging
import os
import signal
import subprocess
import sys
import time
from typing import Dict, Iterator, List, Optional, Type, Union

import psutil
import requests
from agentdesk.util import find_open_port
from mllm import Router
from pydantic import BaseModel
from taskara import Task

from surfkit import config
from surfkit.server.models import V1AgentType, V1SolveTask
from surfkit.types import AgentType
from surfkit.util import find_open_port

from .base import AgentInstance, AgentRuntime, AgentStatus

logger = logging.getLogger(__name__)


class ProcessConnectConfig(BaseModel):
    pass


class ProcessAgentRuntime(AgentRuntime["ProcessAgentRuntime", ProcessConnectConfig]):

    @classmethod
    def name(cls) -> str:
        return "process"

    @classmethod
    def connect_config_type(cls) -> Type[ProcessConnectConfig]:
        return ProcessConnectConfig

    def connect_config(self) -> ProcessConnectConfig:
        return ProcessConnectConfig()

    @classmethod
    def connect(cls, cfg: ProcessConnectConfig) -> "ProcessAgentRuntime":
        return cls()

    def run(
        self,
        agent_type: AgentType,
        name: str,
        version: Optional[str] = None,
        env_vars: Optional[dict] = None,
        llm_providers_local: bool = False,
        owner_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        labels: Optional[Dict[str, str]] = None,
        auth_enabled: bool = True,
        debug: bool = False,
    ) -> AgentInstance:

        port = find_open_port(9090, 10090)
        if not port:
            raise ValueError("Could not find open port")
        logger.debug("running process")

        if not env_vars:
            env_vars = {}

        if llm_providers_local:
            if not agent_type.llm_providers:
                raise ValueError(
                    "No LLM providers in agent type, yet llm_providers_local is True"
                )

            found = {}
            for provider_name in agent_type.llm_providers.preference:
                api_key_env = Router.provider_api_keys.get(provider_name)
                if not api_key_env:
                    raise ValueError(
                        f"No API key environment variable for provider {provider_name}"
                    )
                key = os.getenv(api_key_env)
                if not key:
                    logger.info(
                        f"No API key found locally for provider: {provider_name}"
                    )
                    continue

                logger.info(f"API key found locally for provider: {provider_name}")
                found[api_key_env] = key

            if not found:
                raise ValueError(
                    "No API keys found locally for any of the providers in the agent type"
                )

            env_vars.update(found)

        self.check_llm_providers(agent_type, env_vars)

        command = f"SERVER_PORT={port} nohup {agent_type.cmd} SURFER={name} SERVER_PORT={port} > {os.path.join(config.AGENTSEA_LOG_DIR, name.lower())}.log 2>&1 &"

        os.makedirs(config.AGENTSEA_LOG_DIR, exist_ok=True)
        print(f"running agent on port {port}")

        environment = os.environ.copy()

        if env_vars:
            environment.update(env_vars)  # type: ignore

        environment["AGENT_TYPE"] = agent_type.to_v1().model_dump_json()
        environment["AGENT_NAME"] = name
        environment["SERVER_PORT"] = str(port)

        if not auth_enabled:
            environment["AGENT_NO_AUTH"] = "true"

        if agent_type.llm_providers:
            environment["MODEL_PREFERENCE"] = ",".join(
                agent_type.llm_providers.preference
            )

        if debug:
            environment["DEBUG"] = "true"

        process = subprocess.Popen(
            command,
            shell=True,
            preexec_fn=os.setsid,
            env=environment,
            text=True,
        )

        # Wait for the command to complete
        stdout, stderr = process.communicate()

        # Check if there were any errors
        if process.returncode != 0:
            logger.error("Error running command:")
            print(stderr)
        else:
            # Print the output from stdout
            if stdout:
                print(stdout)

        # Health check logic
        max_retries = 40
        retry_delay = 1
        health_url = f"http://localhost:{port}/health"

        for _ in range(max_retries):
            try:
                response = requests.get(health_url)
                if response.status_code == 200:
                    logger.info("Agent is up and running.")
                    break
            except requests.ConnectionError:
                logger.warning("Agent not yet available, retrying...")
            time.sleep(retry_delay)
        else:
            raise RuntimeError("Failed to start agent, it did not pass health checks.")

        return AgentInstance(
            name=name,
            type=agent_type,
            runtime=self,
            status=AgentStatus.RUNNING,
            port=port,
            labels={"command": command},
            owner_id=owner_id,
        )

    def runtime_local_addr(self, name: str, owner_id: Optional[str] = None) -> str:
        """
        Returns the local address of the agent with respect to the runtime
        """
        instances = AgentInstance.find(name=name, owner_id=owner_id)
        if not instances:
            raise ValueError(f"No instances found for name '{name}'")
        instance = instances[0]

        return f"http://localhost:{instance.port}"

    def solve_task(
        self,
        name: str,
        task: V1SolveTask,
        follow_logs: bool = False,
        attach: bool = False,
        owner_id: Optional[str] = None,
    ) -> None:
        try:
            # Fetch the list of all processes to find the required agent
            process_list = subprocess.check_output(
                f"ps ax -o pid,command | grep -v grep | grep SURFER={name}",
                shell=True,
                text=True,
            )

            if process_list.strip() == "":
                logger.info(f"No running process found with the name {name}.")
                return

            # Extract the port from the process command
            line = process_list.strip().split("\n")[0]
            surf_port = line.split("SERVER_PORT=")[1].split()[0]

            url = f"http://localhost:{surf_port}/v1/tasks"

            task_data = task.model_dump_json()
            headers = {"Content-Type": "application/json"}

            # Send the POST request
            response = requests.post(url, data=task_data, headers=headers)

            # Handle the response
            if response.status_code == 200:
                logger.info("Task successfully posted to the agent.")
                if follow_logs:
                    _task = Task.from_v1(task.task)
                    print("following logs with attach: ", attach)
                    self._follow_logs(name, _task, attach)

            else:
                logger.error(
                    f"Failed to post task: {response.status_code} - {response.text}"
                )

        except subprocess.CalledProcessError as e:
            logger.error("Error while attempting to find the process:", str(e))
        except requests.exceptions.RequestException as e:
            logger.error("Error while sending the POST request:", str(e))
        except Exception as e:
            import traceback

            logger.error(
                f"An unexpected error occurred solving task with runtime {self.name()}: {str(e)} \n{traceback.print_exc()}"
            )

    def _signal_handler(self, agent_name: str):
        def handle_signal(signum, frame):
            print(f"Signal {signum} received, stopping process '{agent_name}'")
            instances = AgentInstance.find(name=agent_name)
            if instances:
                instances[0].delete(force=True)
            else:
                print(f"No instances found for name '{agent_name}'")
            sys.exit(1)

        return handle_signal

    def _follow_logs(self, agent_name: str, task: Task, attach: bool = False):
        log_path = os.path.join(config.AGENTSEA_LOG_DIR, f"{agent_name.lower()}.log")
        if not os.path.exists(log_path):
            logger.error("No log file found.")
            return

        import typer

        with open(log_path, "r") as log_file:
            # Go to the end of the file
            log_file.seek(0, 2)
            try:
                while True:
                    line = log_file.readline()
                    if not line:
                        time.sleep(0.5)  # Wait briefly for new log entries
                        continue
                    clean_line = line.strip()
                    print(clean_line)
                    if clean_line.startswith("► task run ended"):
                        if not attach:
                            print("")
                            stop = typer.confirm(
                                "Task is finished, do you want to stop the agent?"
                            )
                        else:
                            stop = attach

                        if stop:
                            try:
                                instances = AgentInstance.find(name=agent_name)
                                if instances:
                                    instances[0].delete(force=True)
                                else:
                                    print(f"No instances found for name '{agent_name}'")
                            except:
                                pass
                        return
            except KeyboardInterrupt:
                # Handle Ctrl+C gracefully if we are attached to the logs
                print(f"Interrupt received, stopping logs for '{agent_name}'")

                if not attach:
                    print("")
                    stop = typer.confirm("Do you want to stop the agent?")
                else:
                    stop = attach

                if stop:
                    try:
                        instances = AgentInstance.find(name=agent_name)
                        if instances:
                            instances[0].delete(force=True)
                        else:
                            print(f"No instances found for name '{agent_name}'")
                    except:
                        pass
                raise

    def requires_proxy(self) -> bool:
        """Whether this runtime requires a proxy to be used"""
        return False

    def proxy(
        self,
        name: str,
        local_port: Optional[int] = None,
        agent_port: int = 9090,
        background: bool = True,
        owner_id: Optional[str] = None,
    ) -> Optional[int]:
        return

    def get(
        self, name: str, owner_id: Optional[str] = None, source: bool = False
    ) -> AgentInstance:
        if source:
            try:
                # Iterate through all running processes to find the required agent
                for proc in psutil.process_iter(["pid", "environ"]):
                    env_vars = proc.info["environ"]
                    if env_vars and env_vars.get("AGENT_NAME") == name:
                        port = env_vars.get("SERVER_PORT")
                        agent_type_json = env_vars.get("AGENT_TYPE")

                        if port and agent_type_json:
                            agent_type_data = json.loads(agent_type_json)
                            agent_type = AgentType.from_v1(
                                V1AgentType.model_validate(agent_type_data)
                            )
                            return AgentInstance(
                                name=name,
                                type=agent_type,
                                runtime=self,
                                status=AgentStatus.RUNNING,
                                port=int(port),
                            )
                raise ValueError(f"No running process found for agent {name}")
            except Exception as e:
                raise ValueError(f"Error finding process for agent {name}: {str(e)}")
        else:
            instances = AgentInstance.find(
                name=name, owner_id=owner_id, runtime_name=self.name()
            )
            if len(instances) == 0:
                raise ValueError(f"No running agent found with the name {name}")
            return instances[0]

    def list(
        self,
        owner_id: Optional[str] = None,
        source: bool = False,
    ) -> List[AgentInstance]:
        """List agents that are currently running.

        Args:
            owner_id (Optional[str], optional): Owner ID to list for. Defaults to None.
            source (bool, optional): Whether to list from the source. Defaults to False.

        Returns:
            List[AgentInstance]: A list of agent instances
        """
        instances = []

        if source:
            try:
                # Iterate through all running processes
                for proc in psutil.process_iter(["pid", "cmdline", "environ"]):
                    env_vars = proc.info["environ"]
                    if env_vars and "AGENT_NAME" in env_vars:
                        name = env_vars.get("AGENT_NAME")
                        port = env_vars.get("SERVER_PORT")
                        agent_type_json = env_vars.get("AGENT_TYPE")

                        if name and port and agent_type_json:
                            agent_type_data = json.loads(agent_type_json)
                            agent_type = AgentType.from_v1(
                                V1AgentType.model_validate(agent_type_data)
                            )

                            instance = AgentInstance(
                                name=name,
                                type=agent_type,
                                runtime=self,
                                status=AgentStatus.RUNNING,
                                port=int(port),
                            )
                            instances.append(instance)

            except Exception as e:
                logger.error(f"Error processing processes: {str(e)}")
        else:
            instances = AgentInstance.find(owner_id=owner_id, runtime_name=self.name())

        return instances

    def delete(
        self,
        name: str,
        owner_id: Optional[str] = None,
    ) -> None:
        try:
            process_list = subprocess.check_output(
                f"ps ax -o pid,command | grep -v grep | grep SURFER={name}",
                shell=True,
                text=True,
            )
            logger.debug(f"Found process list: {process_list}")
            if process_list.strip():
                # Process found, extract PID and kill it
                pid = process_list.strip().split()[0]
                os.killpg(os.getpgid(int(pid)), signal.SIGTERM)
                logger.info(f"Process {name} with PID {pid} has been terminated.")
            else:
                raise SystemError(f"No running process found with the name {name}.")

            log_file = os.path.join(config.AGENTSEA_LOG_DIR, f"{name}.log")
            if os.path.exists(log_file):
                os.remove(log_file)
                logger.info(f"Deleted log file for {name}.")

        except subprocess.CalledProcessError as e:
            raise SystemError(f"Error while attempting to delete the process: {str(e)}")
        except ValueError as e:
            raise SystemError(f"Error parsing process ID: {str(e)}")
        except Exception as e:
            raise SystemError(f"An unexpected error occurred: {str(e)}")

    def clean(
        self,
        owner_id: Optional[str] = None,
    ) -> None:
        """Clean the runtime by terminating all running processes.

        Args:
            owner_id (Optional[str], optional): Scope to owner ID. Defaults to None.
        """
        try:
            # Fetch the list of all processes that were started with the 'SURFER' environment variable
            process_list = subprocess.check_output(
                "ps ax -o pid,command | grep -v grep | grep SURFER",
                shell=True,
                text=True,
            )
            # Iterate through each process found and kill it
            for line in process_list.strip().split("\n"):
                pid = line.split()[0]  # Extract the PID from the output
                try:
                    os.kill(
                        int(pid), signal.SIGTERM
                    )  # Send SIGTERM signal to terminate the process
                    logger.info(f"Terminated process with PID {pid}.")
                except OSError as e:
                    logger.error(
                        f"Failed to terminate process with PID {pid}: {str(e)}"
                    )
            logger.info("All relevant processes have been terminated.")
        except subprocess.CalledProcessError as e:
            logger.error(
                "No relevant processes found or error executing the ps command:", str(e)
            )
        except Exception as e:
            logger.error(f"An unexpected error occurred during cleanup: {str(e)}")

    def logs(
        self,
        name: str,
        follow: bool = False,
        owner_id: Optional[str] = None,
    ) -> Union[str, Iterator[str]]:
        """
        Retrieve the logs for a specific agent.

        Args:
            name (str): The name of the agent.
            follow (bool, optional): Whether to follow the logs in real-time. Defaults to False.
            owner_id (str, optional): The ID of the owner. Defaults to None.

        Returns:
            Union[str, Iterator[str]]: If `follow` is True, returns an iterator that yields log lines in real-time.
            If `follow` is False, returns all logs as a single string.
            If no logs are available, returns the message "No logs available for this agent."
        """
        log_path = os.path.join(config.AGENTSEA_LOG_DIR, f"{name.lower()}.log")
        if not os.path.exists(log_path):
            return "No logs available for this agent."

        if follow:
            # If follow is True, implement a simple follow (like 'tail -f')
            def follow_logs():
                with open(log_path, "r") as log_file:
                    # Go to the end of the file
                    log_file.seek(0, 2)
                    while True:
                        line = log_file.readline()
                        if not line:
                            time.sleep(0.5)  # Wait briefly for new log entries
                            continue
                        yield line

            return follow_logs()
        else:
            # If not following, return all logs as a single string
            with open(log_path, "r") as log_file:
                return log_file.read()

    def refresh(self, owner_id: Optional[str] = None) -> None:
        """
        Synchronizes the state between running processes and the database.
        Ensures that the processes and the database reflect the same set of running agent instances.

        Parameters:
            owner_id (Optional[str]): The ID of the owner to filter instances.
        """
        # Fetch the running processes
        running_processes_map = {}
        for proc in psutil.process_iter(["pid", "environ"]):
            try:
                env_vars = proc.info["environ"]
                if env_vars and "AGENT_NAME" in env_vars:
                    process_name = env_vars["AGENT_NAME"]
                    running_processes_map[process_name] = env_vars
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                continue

        # Fetch the agent instances from the database
        db_instances = AgentInstance.find(owner_id=owner_id, runtime_name=self.name())

        # Create a mapping of instance names to instances
        db_instances_map = {instance.name: instance for instance in db_instances}

        # Check for processes that are running but not in the database
        for process_name, env_vars in running_processes_map.items():
            if process_name not in db_instances_map:
                print(
                    f"Process '{process_name}' is running but not in the database. Creating new instance."
                )

                port = env_vars.get("SERVER_PORT")
                agent_type_json = env_vars.get("AGENT_TYPE")

                if port and agent_type_json:
                    agent_type_data = json.loads(agent_type_json)
                    agent_type = AgentType.from_v1(
                        V1AgentType.model_validate(agent_type_data)
                    )
                    new_instance = AgentInstance(
                        name=process_name,
                        type=agent_type,
                        runtime=self,
                        status=AgentStatus.RUNNING,
                        port=int(port),
                        owner_id=owner_id,
                    )
                    new_instance.save()

        # Check for instances in the database that are not running as processes
        for instance_name, instance in db_instances_map.items():
            if instance_name not in running_processes_map:
                print(
                    f"Instance '{instance_name}' is in the database but not running. Removing from database."
                )
                instance.delete(force=True)

        logger.debug(
            "Refresh complete. State synchronized between processes and the database."
        )
