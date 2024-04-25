from abc import ABC, abstractmethod
from typing import List, TypeVar, Type, Generic

from pydantic import BaseModel

from devicebay import Device
from taskara import Task

C = TypeVar("C", bound="BaseModel")
T = TypeVar("T", bound="TaskAgent")


class TaskAgent(Generic[C, T], ABC):
    """An agent that works on tasks"""

    @abstractmethod
    def solve_task(
        self,
        task: Task,
        device: Device,
        max_steps: int = 30,
    ) -> Task:
        """Solve a task on a device

        Args:
            task (Task): The task
            device (Device): Device to perform the task on.
            max_steps (int, optional): Max steps allowed. Defaults to 30.

        Returns:
            Task: A task
        """
        pass

    @classmethod
    @abstractmethod
    def supported_devices(cls) -> List[Type[Device]]:
        """Devices this agent supports

        Returns:
            List[Type[Device]]: A list of supported devices
        """
        pass

    @classmethod
    @abstractmethod
    def config_type(cls) -> Type[C]:
        """Type to configure the agent

        Returns:
            Type[C]: A configuration type
        """
        pass

    @classmethod
    @abstractmethod
    def from_config(cls, config: C) -> T:
        """Create an agent from a config

        Args:
            config (C): Config to create the agent from

        Returns:
            T: The Agent
        """
        pass

    @classmethod
    @abstractmethod
    def default(cls) -> T:
        """Create a default agent with no params

        Returns:
            T: The Agent
        """
        pass

    @classmethod
    def init(cls) -> None:
        """Initialize the Agent type"""
        pass
