from .base import AgentTemplate


class SurfSkelly(AgentTemplate):

    def template(self, agent_name: str) -> str:

        return f"""
from typing import List, Type, Tuple, Optional
import logging
from typing import Final
import traceback
import time
import os

from devicebay import Device
from rich.console import Console
from pydantic import BaseModel
from surfkit.agent import TaskAgent
from taskara import Task
from mllm import Router

logging.basicConfig(level=logging.INFO)
logger: Final = logging.getLogger(__name__)
logger.setLevel(int(os.getenv("LOG_LEVEL", str(logging.DEBUG))))

console = Console(force_terminal=True)

router = Router.from_env()


class {agent_name}Config(BaseModel):
    pass


class {agent_name}(TaskAgent):
    \"""A desktop agent that uses GPT-4V augmented with OCR and Grounding Dino to solve tasks\"""

    def solve_task(
        self,
        task: Task,
        device: Optional[Device] = None,
        max_steps: int = 30,
    ) -> Task:
        \"""Solve a task

        Args:
            task (Task): Task to solve.
            device (Device, optional): Device to perform the task on. Defaults to None.
            max_steps (int, optional): Max steps to try and solve. Defaults to 30.

        Returns:
            Task: The task
        \"""

        # <SOLVE TASK HERE>
        pass

    @classmethod
    def supported_devices(cls) -> List[Type[Device]]:
        \"""Devices this agent supports

        Returns:
            List[Type[Device]]: A list of supported devices
        \"""
        return [Desktop]

    @classmethod
    def config_type(cls) -> Type[{agent_name}Config]:
        \"""Type of config

        Returns:
            Type[DinoConfig]: Config type
        \"""
        return {agent_name}Config

    @classmethod
    def from_config(cls, config: {agent_name}Config) -> "{agent_name}":
        \"""Create an agent from a config

        Args:
            config (DinoConfig): Agent config

        Returns:
            {agent_name}: The agent
        \"""
        return {agent_name}()

    @classmethod
    def default(cls) -> "{agent_name}":
        \"""Create a default agent

        Returns:
            {agent_name}: The agent
        \"""
        return {agent_name}()

    @classmethod
    def init(cls) -> None:
        \"""Initialize the agent class\"""
        # <INITIALIZE AGENT HERE>
        return


Agent = SurfPizza

"""
