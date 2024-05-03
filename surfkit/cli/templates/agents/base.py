from abc import ABC, abstractmethod


class AgentTemplate(ABC):
    @abstractmethod
    def template(self, agent_name: str) -> str:
        pass
