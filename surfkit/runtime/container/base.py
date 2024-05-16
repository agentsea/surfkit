from abc import ABC, abstractmethod
from typing import Generic, List, Optional, Tuple, Type, TypeVar

from pydantic import BaseModel

R = TypeVar("R", bound="ContainerRuntime")
C = TypeVar("C", bound="BaseModel")


class ContainerRuntime(Generic[C, R], ABC):

    @classmethod
    def name(cls) -> str:
        return cls.__name__

    @classmethod
    @abstractmethod
    def connect_config_type(cls) -> Type[C]:
        pass

    @classmethod
    @abstractmethod
    def connect(cls, cfg: C) -> R:
        pass

    @abstractmethod
    def create(
        self,
        image: str,
        name: Optional[str] = None,
        env_vars: Optional[dict] = None,
        mem_request: Optional[str] = "500m",
        mem_limit: Optional[str] = "2Gi",
        cpu_request: Optional[str] = "1",
        cpu_limit: Optional[str] = "4",
        gpu_mem: Optional[str] = None,
    ) -> None:
        pass

    @abstractmethod
    def call(
        self,
        name: str,
        path: str,
        method: str,
        port: int = 8080,
        data: Optional[dict] = None,
        headers: Optional[dict] = None,
    ) -> Tuple[int, str]:
        pass

    @abstractmethod
    def delete(self, name: str) -> None:
        pass

    @abstractmethod
    def list(self) -> List[str]:
        pass

    @abstractmethod
    def clean(self) -> None:
        pass

    @abstractmethod
    def logs(self, name: str) -> str:
        pass
