from abc import ABC, abstractmethod

from agentdesk import Desktop
from taskara import Task


class TaskAgent(ABC):
    """An agent that works on tasks"""

    @abstractmethod
    def solve_task(
        self,
        task: Task,
        desktop: Desktop,
        max_steps: int = 10,
    ) -> Task:
        """Solve a desktop GUI task

        Args:
            task (Task): The task
            max_steps (int, optional): Max steps allowed. Defaults to 10.
            site_url (Optional[str], optional): A starting site. Defaults to None.

        Returns:
            Task: A task
        """
        pass
