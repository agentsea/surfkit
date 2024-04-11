from typing import List, Optional, Tuple, TypeVar, Type, Generic
from abc import ABC, abstractmethod

from pydantic import BaseModel


R = TypeVar("R", bound="ContainerRuntime")
C = TypeVar("C", bound="BaseModel")


class ContainerRuntime(Generic[C, R], ABC):

    @abstractmethod
    @classmethod
    def connect_config_type(cls) -> Type[C]:
        pass

    @abstractmethod
    @classmethod
    def connect(cls, cfg: C) -> R:
        pass

    @abstractmethod
    def create(self, image: str, name: Optional[str] = None) -> None:
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
