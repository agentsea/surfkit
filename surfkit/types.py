import json
import logging
import os
import time
import uuid
from typing import Any, Dict, List, Optional

import requests
import yaml
from sqlalchemy import or_

from surfkit.config import GlobalConfig

from .db.conn import WithDB
from .db.models import AgentTypeRecord
from .env import HUB_API_KEY_ENV, HUB_SERVER_API_ENV
from .server.models import (V1AgentType, V1AgentTypes, V1DeviceConfig,
                            V1EnvVarOpt, V1Find, V1LLMProviders, V1Meter,
                            V1ResourceLimits, V1ResourceRequests, V1Runtime)

logger = logging.getLogger(__name__)


class AgentType(WithDB):
    """A type of agent"""

    def __init__(
        self,
        name: str,
        description: str,
        kind: str,
        cmd: str,
        img_repo: str,
        versions: Dict[str, str],
        runtimes: List[V1Runtime] = [],
        owner_id: Optional[str] = None,
        env_opts: List[V1EnvVarOpt] = [],
        public: bool = False,
        icon: Optional[str] = None,
        resource_requests: V1ResourceRequests = V1ResourceRequests(),
        resource_limits: V1ResourceLimits = V1ResourceLimits(),
        llm_providers: Optional[V1LLMProviders] = None,
        devices: List[V1DeviceConfig] = [],
        repo: Optional[str] = None,
        meters: List[V1Meter] = [],
        remote: Optional[str] = None,
        tags: List[str] = [],
        labels: Dict[str, str] = {},
    ):
        self.id = str(uuid.uuid4())
        self.name = name
        self.description = description
        self.kind = kind
        self.cmd = cmd
        self.img_repo = img_repo
        self.versions = versions
        self.runtimes = runtimes
        self.owner_id = owner_id
        self.env_opts = env_opts
        self.public = public
        self.icon = icon
        self.resource_requests = resource_requests
        self.resource_limits = resource_limits
        self.created = time.time()
        self.updated = time.time()
        self.llm_providers: Optional[V1LLMProviders] = llm_providers
        self.devices = devices
        self.repo = repo
        self.meters = meters
        self.tags = tags
        self.labels = labels
        self.remote = remote
        self.save()

    @classmethod
    def from_file(
        cls, path: str = "./agent.yaml", owner_id: Optional[str] = None
    ) -> "AgentType":
        with open(path) as f:
            data = yaml.safe_load(f)

        v1 = V1AgentType.model_validate(data)
        return cls.from_v1(v1, owner_id)

    def to_v1(self) -> V1AgentType:
        return V1AgentType(
            id=self.id,
            name=self.name,
            description=self.description,
            kind=self.kind,
            cmd=self.cmd,
            img_repo=self.img_repo,
            versions=self.versions,
            env_opts=self.env_opts,
            runtimes=self.runtimes,
            created=self.created,
            updated=self.updated,
            public=self.public,
            icon=self.icon,
            resource_requests=self.resource_requests,
            resource_limits=self.resource_limits,
            llm_providers=self.llm_providers,
            devices=self.devices,
            owner_id=self.owner_id,
            repo=self.repo,
            meters=self.meters,
            tags=self.tags,
            labels=self.labels,
        )

    @classmethod
    def from_v1(cls, v1: V1AgentType, owner_id: Optional[str] = None) -> "AgentType":
        obj = cls.__new__(cls)
        obj.id = v1.id
        obj.name = v1.name
        obj.kind = v1.kind
        obj.owner_id = v1.owner_id
        obj.description = v1.description
        obj.cmd = v1.cmd
        obj.img_repo = v1.img_repo
        obj.env_opts = v1.env_opts
        obj.runtimes = v1.runtimes
        obj.created = v1.created
        obj.updated = v1.updated
        obj.public = v1.public
        obj.icon = v1.icon
        obj.resource_requests = v1.resource_requests
        obj.resource_limits = v1.resource_limits
        obj.versions = v1.versions
        obj.llm_providers = v1.llm_providers
        obj.devices = v1.devices
        obj.repo = v1.repo
        obj.meters = v1.meters
        obj.owner_id = owner_id
        obj.tags = v1.tags
        obj.labels = v1.labels
        return obj

    def to_record(self) -> AgentTypeRecord:
        versions = json.dumps(self.versions)
        llm_providers = None
        if self.llm_providers:
            llm_providers = json.dumps(self.llm_providers.model_dump())

        meters = None
        if self.meters:
            meters = json.dumps([meter.model_dump() for meter in self.meters])

        devices = None
        if self.devices:
            devices = json.dumps(self.devices)
        return AgentTypeRecord(
            id=self.id,
            name=self.name,
            description=self.description,
            kind=self.kind,
            cmd=self.cmd,
            img_repo=self.img_repo,
            versions=versions,
            env_opts=json.dumps([opt.model_dump() for opt in self.env_opts]),
            runtimes=json.dumps([runtime.model_dump() for runtime in self.runtimes]),
            created=self.created,
            updated=self.updated,
            owner_id=self.owner_id,
            public=self.public,
            icon=self.icon,
            resource_limits=json.dumps(self.resource_limits.model_dump()),
            resource_requests=json.dumps(self.resource_requests.model_dump()),
            llm_providers=llm_providers,
            devices=devices,
            meters=meters,
            repo=self.repo,
            tags=json.dumps(self.tags),
            labels=json.dumps(self.labels),
        )

    @classmethod
    def from_record(cls, record: AgentTypeRecord) -> "AgentType":
        versions = {}
        if record.versions:  # type: ignore
            versions = json.loads(str(record.versions))

        llm_providers = None
        if record.llm_providers:  # type: ignore
            llm_providers = V1LLMProviders(**json.loads(str(record.llm_providers)))

        devices = []
        if record.devices:  # type: ignore
            devices = json.loads(str(record.devices))

        meters = []
        if record.meters:  # type: ignore
            meters_mod = json.loads(str(record.meters))
            meters = [V1Meter(**m) for m in meters_mod]

        obj = cls.__new__(cls)
        obj.id = record.id
        obj.name = record.name
        obj.kind = record.kind
        obj.description = record.description
        obj.cmd = record.cmd
        obj.img_repo = record.img_repo
        obj.versions = versions
        obj.env_opts = [V1EnvVarOpt(**opt) for opt in json.loads(str(record.env_opts))]
        obj.runtimes = [
            V1Runtime(**runtime) for runtime in json.loads(str(record.runtimes))
        ]
        obj.created = record.created
        obj.updated = record.updated
        obj.owner_id = record.owner_id
        obj.public = record.public
        obj.icon = record.icon
        obj.resource_requests = (
            V1ResourceRequests(**json.loads(str(record.resource_requests)))
            if record.resource_requests  # type: ignore
            else V1ResourceRequests()
        )
        obj.resource_limits = (
            V1ResourceLimits(**json.loads(str(record.resource_limits)))
            if record.resource_limits  # type: ignore
            else V1ResourceLimits()
        )
        obj.llm_providers = llm_providers
        obj.devices = devices
        obj.meters = meters
        obj.repo = record.repo
        obj.tags = json.loads(str(record.tags))
        obj.labels = json.loads(str(record.labels))
        return obj

    def save(self) -> None:
        for session in self.get_db():
            if session:
                record = self.to_record()
                session.merge(record)
                session.commit()

    @classmethod
    def find(cls, remote: Optional[str] = None, **kwargs) -> List["AgentType"]:
        if remote:
            logger.debug(
                "finding remote agent_types for: ", remote, kwargs.get("owner_id")
            )
            remote_response = cls._remote_request(
                remote,
                "GET",
                "/v1/agenttypes",
                json_data=V1Find(args=kwargs).model_dump(),
            )
            agent_types = V1AgentTypes(**remote_response)
            if remote_response is not None:
                out = [
                    cls.from_v1(record, kwargs.get("owner_id"))
                    for record in agent_types.types
                ]
                for type in out:
                    type.remote = remote
                    logger.debug("returning type: ", type.__dict__)
                return out
            else:
                return []
        else:
            for session in cls.get_db():
                records = session.query(AgentTypeRecord).filter_by(**kwargs).all()
                return [cls.from_record(record) for record in records]

            return []

    @classmethod
    def find_for_user(
        cls, user_id: str, name: Optional[str] = None
    ) -> List["AgentType"]:
        for session in cls.get_db():
            # Base query
            query = session.query(AgentTypeRecord).filter(
                or_(
                    AgentTypeRecord.owner_id == user_id,  # type: ignore
                    AgentTypeRecord.public == True,
                )
            )

            # Conditionally add name filter if name is provided
            if name is not None:
                query = query.filter(AgentTypeRecord.name == name)

            records = query.all()
            return [cls.from_record(record) for record in records]
        return []

    @classmethod
    def delete(cls, name: str, owner_id: str) -> None:
        for session in cls.get_db():
            if session:
                record = (
                    session.query(AgentTypeRecord)
                    .filter_by(name=name, owner_id=owner_id)
                    .first()
                )
                if record:
                    session.delete(record)
                    session.commit()

    def remove(self) -> None:
        for session in self.get_db():
            if session:
                record = (
                    session.query(AgentTypeRecord)
                    .filter_by(name=self.name, owner_id=self.owner_id)
                    .first()
                )
                if record:
                    session.delete(record)
                    session.commit()

    def update(self, model: V1AgentType) -> None:
        """
        Updates the current AgentType instance with values from an AgentTypeModel instance.
        """
        # Track if any updates are needed
        updated = False

        if self.description != model.description:
            self.description = model.description
            updated = True

        if self.kind != model.kind:
            self.kind = model.kind
            updated = True

        if self.cmd != model.cmd:
            self.cmd = model.cmd
            updated = True

        if self.img_repo != model.img_repo:
            self.img_repo = model.img_repo
            updated = True

        if self.versions != model.versions:
            self.versions = model.versions
            updated = True

        if self.env_opts != model.env_opts:
            self.env_opts = model.env_opts
            updated = True

        if self.runtimes != model.runtimes:
            self.runtimes = model.runtimes
            updated = True

        if self.public != model.public:
            self.public = model.public
            updated = True

        if self.icon != model.icon:
            self.icon = model.icon
            updated = True

        if self.resource_requests != model.resource_requests:
            self.resource_requests = model.resource_requests
            updated = True

        if self.resource_limits != model.resource_limits:
            self.resource_limits = model.resource_limits
            updated = True

        if self.llm_providers != model.llm_providers:
            self.llm_providers = model.llm_providers
            updated = True

        if self.devices != model.devices:
            self.devices = model.devices
            updated = True

        if self.meters != model.meters:
            self.meters = model.meters
            updated = True

        if self.repo != model.repo:
            self.repo = model.repo
            updated = True

        if self.tags != model.tags:
            self.tags = model.tags
            updated = True

        if self.labels != model.labels:
            self.labels = model.labels
            updated = True

        # If anything was updated, set the updated timestamp and save changes
        if updated:
            self.updated = time.time()
            self.save()

    @classmethod
    def _remote_request(
        cls,
        addr: str,
        method: str,
        endpoint: str,
        json_data: Optional[dict] = None,
        auth_token: Optional[str] = None,
    ) -> Any:
        url = f"{addr}{endpoint}"
        headers = {}
        if not auth_token:
            auth_token = os.getenv(HUB_API_KEY_ENV)
            if not auth_token:
                config = GlobalConfig.read()
                if not config.api_key:
                    raise ValueError(
                        "No API key found. Please run `surfkit login` first."
                    )
                auth_token = config.api_key
        logger.debug(f"auth_token: {auth_token}")
        headers["Authorization"] = f"Bearer {auth_token}"
        try:
            if method.upper() == "GET":
                logger.debug("\ncalling remote task GET with url: ", url)
                logger.debug("\ncalling remote task GET with headers: ", headers)
                response = requests.get(url, headers=headers)
            elif method.upper() == "POST":
                logger.debug("\ncalling remote task POST with: ", url)
                logger.debug("\ncalling remote task POST with headers: ", headers)
                response = requests.post(url, json=json_data, headers=headers)
            elif method.upper() == "PUT":
                logger.debug("\ncalling remote task PUT with: ", url)
                logger.debug("\ncalling remote task PUT with headers: ", headers)
                response = requests.put(url, json=json_data, headers=headers)
            elif method.upper() == "DELETE":
                logger.debug("\ncalling remote task DELETE with: ", url)
                logger.debug("\ncalling remote task DELETE with headers: ", headers)
                response = requests.delete(url, headers=headers)
            else:
                return None

            try:
                response.raise_for_status()
            except requests.exceptions.HTTPError as e:
                print("HTTP Error:", e)
                print("Status Code:", response.status_code)
                try:
                    print("Response Body:", response.json())
                except ValueError:
                    print("Raw Response:", response.text)
                raise
            logger.debug("\nresponse: ", response.__dict__)
            logger.debug("\response.status_code: ", response.status_code)

            try:
                response_json = response.json()
                logger.debug("\nresponse_json: ", response_json)
                return response_json
            except ValueError:
                print("Raw Response:", response.text)
                return None

        except requests.RequestException as e:
            raise e
