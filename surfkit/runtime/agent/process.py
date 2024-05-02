from typing import List, Optional, Type, Union, Iterator
import os
import subprocess
import time
import signal
import json
import logging

from taskara.server.models import V1Task
from agentdesk.util import find_open_port
import requests
from pydantic import BaseModel
from mllm import Router

from .base import AgentRuntime, AgentInstance
from surfkit.models import V1AgentType, V1SolveTask
from surfkit.types import AgentType
from surfkit.util import find_open_port


logger = logging.getLogger(__name__)


class ConnectConfig(BaseModel):
    pass


class ProcessAgentRuntime(AgentRuntime):

    @classmethod
    def name(cls) -> str:
        return "process"

    @classmethod
    def connect_config_type(cls) -> Type[ConnectConfig]:
        return ConnectConfig

    @classmethod
    def connect(cls, cfg: ConnectConfig) -> "ProcessAgentRuntime":
        return cls()

    def run(
        self,
        agent_type: AgentType,
        name: str,
        version: Optional[str] = None,
        env_vars: Optional[dict] = None,
        llm_providers_local: bool = False,
        owner_id: Optional[str] = None,
    ) -> AgentInstance:

        port = find_open_port(9090, 9990)
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

        # Create metadata directory if it does not exist
        os.makedirs(f".data/proc", exist_ok=True)
        # Write metadata to a file
        with open(f".data/proc/{name}.json", "w") as f:
            json.dump(metadata, f, indent=4)

        os.makedirs(f".data/logs", exist_ok=True)
        command = f"nohup {agent_type.cmd} SURFER={name} SURF_PORT={port}> ./.data/logs/{name.lower()}.log 2>&1 &"

        process = subprocess.Popen(
            command,
            shell=True,
            preexec_fn=os.setsid,
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
            print(stdout)

        return AgentInstance(name, agent_type, self, port)

    def solve_task(
        self, agent_name: str, task: V1SolveTask, follow_logs: bool = False
    ) -> None:
        try:
            # Fetch the list of all processes to find the required agent
            process_list = subprocess.check_output(
                f"ps ax -o pid,command | grep -v grep | grep SURFER={agent_name}",
                shell=True,
                text=True,
            )

            if process_list.strip() == "":
                logger.info(f"No running process found with the name {agent_name}.")
                return

            # Extract the port from the process command
            line = process_list.strip().split("\n")[0]
            surf_port = line.split("SURF_PORT=")[1].split()[0]

            # Prepare the URL for the POST request
            url = f"http://localhost:{surf_port}/v1/tasks"

            # Convert the task model to JSON
            task_data = task.model_dump_json()

            # Headers for the POST request
            headers = {"Content-Type": "application/json"}

            # Send the POST request
            response = requests.post(url, data=task_data, headers=headers)

            # Handle the response
            if response.status_code == 200:
                logger.info("Task successfully posted to the agent.")
                if follow_logs:
                    # If required, follow the logs
                    logs = self.logs(agent_name, follow=True)
                    for log in logs:
                        print(log)
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

    def proxy(
        self,
        name: str,
        local_port: Optional[int] = None,
        pod_port: int = 9090,
        background: bool = True,
    ) -> None:
        logger.info("no proxy needed")
        return

    def get(self, name: str) -> AgentInstance:
        try:
            # Read the metadata file
            with open(f".data/proc/{name}.json", "r") as f:
                metadata = json.load(f)

            agent_type = V1AgentType.model_validate(metadata["agent_type"])
            return AgentInstance(
                metadata["name"], AgentType.from_v1(agent_type), self, metadata["port"]
            )
        except FileNotFoundError:
            raise ValueError(f"No metadata found for agent {name}")

    def list(self) -> List[AgentInstance]:
        instances = []
        metadata_dir = ".data/proc"
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
                        agent_type = V1AgentType.model_validate(metadata["agent_type"])
                        standard_agent_type = AgentType.from_v1(agent_type)
                        instance = AgentInstance(
                            name=metadata["name"],
                            type=standard_agent_type,
                            runtime=self,
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

        return instances

    def delete(self, name: str) -> None:
        try:
            process_list = subprocess.check_output(
                f"ps ax -o pid,command | grep -v grep | grep SURFER={name}",
                shell=True,
                text=True,
            )
            if process_list.strip():
                # Process found, extract PID and kill it
                pid = process_list.strip().split()[0]
                os.kill(int(pid), signal.SIGTERM)
                logger.info(f"Process {name} with PID {pid} has been terminated.")
            else:
                logger.info(f"No running process found with the name {name}.")

            # Delete the metadata file whether or not the process was found
            metadata_file = f".data/proc/{name}.json"
            if os.path.exists(metadata_file):
                os.remove(metadata_file)
                logger.info(f"Deleted metadata file for {name}.")

        except subprocess.CalledProcessError as e:
            logger.error("Error while attempting to delete the process:", str(e))
        except ValueError as e:
            logger.error("Error parsing process ID:", str(e))
        except Exception as e:
            logger.error(f"An unexpected error occurred: {str(e)}")

    def clean(self) -> None:
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

    def logs(self, name: str, follow: bool = False) -> Union[str, Iterator[str]]:
        log_path = f"./.data/logs/{name.lower()}.log"
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
