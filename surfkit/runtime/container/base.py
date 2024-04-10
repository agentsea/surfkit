from typing import List, Optional
from abc import ABC, abstractmethod

import requests


class ConatinerRuntime(ABC):
    @abstractmethod
    def create(self, image: str, name: Optional[str] = None) -> None:
        pass

    @abstractmethod
    def call(
        self,
        name: str,
        route: str,
        method: str = "GET",
        port: int = 8080,
        params: Optional[dict] = None,
        body: Optional[dict] = None,
        headers: Optional[dict] = None,
    ) -> requests.Response:
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
