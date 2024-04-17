from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Optional
import os
import yaml

from .env import AGENTSEA_HUB_URL_ENV, HUB_SERVER_ENV, HUB_SERVER_API_ENV

import rootpath

AGENTSEA_HUB_URL = os.getenv(AGENTSEA_HUB_URL_ENV, "https://hub.agentsea.ai")
HUB_URL = os.getenv(HUB_SERVER_ENV, "https://surf.agentlabs.xyz")
HUB_API_URL = os.getenv(HUB_SERVER_API_ENV, "https://api.surf.agentlabs.xyz")


@dataclass
class GlobalConfig:
    api_key: Optional[str] = None
    hub_address: str = AGENTSEA_HUB_URL

    def write(self) -> None:
        home = os.path.expanduser("~")
        dir = os.path.join(home, ".agentsea")
        os.makedirs(dir, exist_ok=True)
        path = os.path.join(dir, "config.yaml")

        with open(path, "w") as yaml_file:
            yaml.dump(self.__dict__, yaml_file)
            yaml_file.flush()
            yaml_file.close()

    @classmethod
    def read(cls) -> GlobalConfig:
        home = os.path.expanduser("~")
        dir = os.path.join(home, ".agentsea")
        os.makedirs(dir, exist_ok=True)
        path = os.path.join(dir, "config.yaml")

        if not os.path.exists(path):
            return GlobalConfig()

        with open(path, "r") as yaml_file:
            config = yaml.safe_load(yaml_file)
            return GlobalConfig(**config)


@dataclass
class Config:
    name: str
    summary: str
    description: str

    @classmethod
    def from_project(cls) -> Config:
        path = rootpath.detect()
        if not path:
            raise SystemError("could not detect root python path")

        config_path = os.path.join(path, "agent.yaml")

        if not os.path.exists(config_path):
            raise SystemError("could not detect agent.yaml in project root")

        with open(config_path, "r") as yaml_file:
            config = yaml.safe_load(yaml_file)

            return Config(
                name=config["name"],
                summary=config["summary"],
                description=config["description"],
            )
