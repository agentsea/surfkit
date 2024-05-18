import json
import logging
import os
import signal
import subprocess
import sys
import time
from typing import Dict, Iterator, List, Optional, Type, Union

import requests
from agentdesk.util import find_open_port
from mllm import Router
from pydantic import BaseModel

from surfkit import config
from surfkit.server.models import V1AgentType, V1SolveTask
from surfkit.types import AgentType
from surfkit.util import find_open_port

from .base import AgentInstance, AgentRuntime

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
    ) -> AgentInstance:

        port = find_open_port(9090, 10090)
        if not port:
            raise ValueError("Could not find open port")
        logger.debug("running process")

        metadata = {
            "name": name,
            "agent_type": agent_type.to_v1().model_dump(),
            "port": port,
            "env_vars": env_vars if env_vars else {},
            "version": version,
            "owner_id": owner_id,
        }

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
            metadata["env_vars"].update(found)

        command = f"SURF_PORT={port} nohup {agent_type.cmd} SURFER={name} SURF_PORT={port} > {os.path.join(config.AGENTSEA_LOG_DIR, name.lower())}.log 2>&1 &"
        metadata["command"] = command

        # Create metadata directory if it does not exist
        os.makedirs(config.AGENTSEA_PROC_DIR, exist_ok=True)
        # Write metadata to a file
        with open(os.path.join(config.AGENTSEA_PROC_DIR, f"{name}.json"), "w") as f:
            json.dump(metadata, f, indent=4)

        os.makedirs(config.AGENTSEA_LOG_DIR, exist_ok=True)
        print(f"running agent on port {port}")

        environment = os.environ.copy()
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
        max_retries = 30
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
            status="running",
            port=port,
            labels={"command": command},
            owner_id=owner_id,
        )

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
            surf_port = line.split("SURF_PORT=")[1].split()[0]

            url = f"http://localhost:{surf_port}/v1/tasks"

            task_data = task.model_dump_json()
            headers = {"Content-Type": "application/json"}

            # Send the POST request
            response = requests.post(url, data=task_data, headers=headers)

            # Handle the response
            if response.status_code == 200:
                logger.info("Task successfully posted to the agent.")
                if follow_logs:
                    # If required, follow the logs
                    if attach:
                        signal.signal(signal.SIGINT, self._signal_handler(name))
                    self._follow_logs(name)

            else:
                logger.error(
                    f"Failed to post task: {response.status_code} - {response.text}"
                )

        except subprocess.CalledProcessError as e:
            logger.error("Error while attempting to find the process:", str(e))
        except requests.exceptions.RequestException as e:
            logger.error("Error while sending the POST request:", str(e))
        except Exception as e:
            logger.error(f"An unexpected error occurred: {str(e)}")

    def _signal_handler(self, agent_name: str):
        def handle_signal(signum, frame):
            print(f"Signal {signum} received, stopping process '{agent_name}'")
            self.delete(agent_name)
            instances = AgentInstance.find(name=agent_name)
            if instances:
                instances[0].delete()
            sys.exit(1)

        return handle_signal

    def _follow_logs(self, agent_name: str):
        log_path = os.path.join(config.AGENTSEA_LOG_DIR, f"{agent_name.lower()}.log")
        if not os.path.exists(log_path):
            logger.error("No log file found.")
            return

        with open(log_path, "r") as log_file:
            # Go to the end of the file
            log_file.seek(0, 2)
            try:
                while True:
                    line = log_file.readline()
                    if not line:
                        time.sleep(0.5)  # Wait briefly for new log entries
                        continue
                    print(line.strip())
            except KeyboardInterrupt:
                # Handle Ctrl+C gracefully if we are attached to the logs
                print(f"Interrupt received, stopping logs for '{agent_name}'")
                self.delete(agent_name)
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
        logger.info("no proxy needed")
        return

    def get(
        self, name: str, owner_id: Optional[str] = None, source: bool = False
    ) -> AgentInstance:
        if source:
            try:
                # Read the metadata file
                with open(
                    os.path.join(config.AGENTSEA_PROC_DIR, f"{name}.json", "r")
                ) as f:
                    metadata = json.load(f)

                agent_type = V1AgentType.model_validate(metadata["agent_type"])
                return AgentInstance(
                    metadata["name"],
                    AgentType.from_v1(agent_type),
                    self,
                    metadata["port"],
                )
            except FileNotFoundError:
                raise ValueError(f"No metadata found for agent {name}")

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
        instances = []
        if source:
            metadata_dir = config.AGENTSEA_PROC_DIR
            all_processes = subprocess.check_output(
                "ps ax -o pid,command", shell=True, text=True
            )

            for filename in os.listdir(metadata_dir):
                if filename.endswith(".json"):
                    try:
                        with open(os.path.join(metadata_dir, filename), "r") as file:
                            metadata = json.load(file)

                        # Check if process is still running
                        process_info = f"SURFER={metadata['name']} "
                        if process_info in all_processes:
                            agent_type = V1AgentType.model_validate(
                                metadata["agent_type"]
                            )
                            standard_agent_type = AgentType.from_v1(agent_type)
                            instance = AgentInstance(
                                name=metadata["name"],
                                type=standard_agent_type,
                                runtime=self,
                                status="running",
                                port=metadata["port"],
                            )
                            instances.append(instance)
                        else:
                            # Process is not running, delete the metadata file
                            os.remove(os.path.join(metadata_dir, filename))
                            logger.info(
                                f"Deleted metadata for non-existing process {metadata['name']}."
                            )

                    except Exception as e:
                        logger.error(f"Error processing {filename}: {str(e)}")
        else:
            return AgentInstance.find(owner_id=owner_id, runtime_name=self.name())

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

            # Delete the metadata file whether or not the process was found
            metadata_file = os.path.join(config.AGENTSEA_PROC_DIR, f"{name}.json")
            if os.path.exists(metadata_file):
                os.remove(metadata_file)
                logger.info(f"Deleted metadata file for {name}.")

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
