from abc import ABC, abstractmethod
from typing import List

from devicebay import Device
from taskara import Task


class TaskAgent(ABC):
    """An agent that works on tasks"""

    @abstractmethod
    def solve_task(
        self,
        task: Task,
        device: Device,
        max_steps: int = 10,
    ) -> Task:
        """Solve a task on a device

        Args:
            task (Task): The task
            device (Desktop): Device to perform the task on.
            max_steps (int, optional): Max steps allowed. Defaults to 10.

        Returns:
            Task: A task
        """
        pass

    @classmethod
    @abstractmethod
    def supported_devices(cls) -> List[str]:
        """Devices this agent supports

        Returns:
            List[str]: A list of supported devices
        """
        pass
