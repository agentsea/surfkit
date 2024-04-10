from abc import ABC, abstractmethod


class VM:
    pass


class VMRuntime(ABC):
    """A virtual machine runntime"""

    @abstractmethod
    def create(self) -> VM:
        pass

    @abstractmethod
    def delete(self, name: str) -> None:
        pass

    @abstractmethod
    def stop(self, name: str) -> None:
        pass

    @abstractmethod
    def start(self, name: str) -> None:
        pass
