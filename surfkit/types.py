from typing import List, Optional, Dict
import uuid
import time
import json

from sqlalchemy import or_

from .db.models import AgentTypeRecord
from .db.conn import WithDB
from .models import (
    EnvVarOptModel,
    AgentTypeModel,
    LLMProviders,
    LLMProviders,
    DeviceConfig,
    RuntimeModel,
    MeterModel,
    ResourceLimitsModel,
    ResourceRequestsModel,
)


class AgentType(WithDB):
    """A type of agent"""

    def __init__(
        self,
        name: str,
        description: str,
        image: str,
        versions: Dict[str, str],
        runtimes: List[RuntimeModel] = [],
        owner_id: Optional[str] = None,
        env_opts: List[EnvVarOptModel] = [],
        public: bool = False,
        icon: Optional[str] = None,
        resource_requests: ResourceRequestsModel = ResourceRequestsModel(),
        resource_limits: ResourceLimitsModel = ResourceLimitsModel(),
        llm_providers: Optional[LLMProviders] = None,
        devices: List[DeviceConfig] = [],
        repo: Optional[str] = None,
        meters: List[MeterModel] = [],
    ):
        self.id = str(uuid.uuid4())
        self.name = name
        self.description = description
        self.image = image
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
        self.llm_providers: Optional[LLMProviders] = llm_providers
        self.devices = devices
        self.repo = repo
        self.meters = meters
        self.save()

    def to_schema(self) -> AgentTypeModel:
        return AgentTypeModel(
            id=self.id,
            name=self.name,
            description=self.description,
            image=self.image,
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
        )

    @classmethod
    def from_schema(cls, schema: AgentTypeModel) -> "AgentType":
        obj = cls.__new__(cls)
        obj.id = schema.id
        obj.name = schema.name
        obj.owner_id = schema.owner_id
        obj.description = schema.description
        obj.image = schema.image
        obj.env_opts = schema.env_opts
        obj.runtimes = schema.runtimes
        obj.created = schema.created
        obj.updated = schema.updated
        obj.public = schema.public
        obj.icon = schema.icon
        obj.resource_requests = schema.resource_requests
        obj.resource_limits = schema.resource_limits
        obj.versions = schema.versions
        obj.llm_providers = schema.llm_providers
        obj.devices = schema.devices
        obj.repo = schema.repo
        obj.meters = schema.meters
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
            image=self.image,
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
        )

    @classmethod
    def from_record(cls, record: AgentTypeRecord) -> "AgentType":
        versions = {}
        if record.versions:  # type: ignore
            versions = json.loads(str(record.versions))

        llm_providers = None
        if record.llm_providers:  # type: ignore
            llm_providers = LLMProviders(**json.loads(str(record.llm_providers)))

        devices = []
        if record.devices:  # type: ignore
            devices = json.loads(str(record.devices))

        meters = []
        if record.meters:  # type: ignore
            meters_mod = json.loads(str(record.meters))
            meters = [MeterModel(**m) for m in meters_mod]

        obj = cls.__new__(cls)
        obj.id = record.id
        obj.name = record.name
        obj.description = record.description
        obj.image = record.image
        obj.versions = versions
        obj.env_opts = [
            EnvVarOptModel(**opt) for opt in json.loads(str(record.env_opts))
        ]
        obj.runtimes = [
            RuntimeModel(**runtime) for runtime in json.loads(str(record.runtimes))
        ]
        obj.created = record.created
        obj.updated = record.updated
        obj.owner_id = record.owner_id
        obj.public = record.public
        obj.icon = record.icon
        obj.resource_requests = (
            ResourceRequestsModel(**json.loads(str(record.resource_requests)))
            if record.resource_requests  # type: ignore
            else ResourceRequestsModel()
        )
        obj.resource_limits = (
            ResourceLimitsModel(**json.loads(str(record.resource_limits)))
            if record.resource_limits  # type: ignore
            else ResourceLimitsModel()
        )
        obj.llm_providers = llm_providers
        obj.devices = devices
        obj.meters = meters
        obj.repo = record.repo
        return obj

    def save(self) -> None:
        for session in self.get_db():
            if session:
                record = self.to_record()
                session.merge(record)
                session.commit()

    @classmethod
    def find(cls, **kwargs) -> List["AgentType"]:
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

    def update(self, model: AgentTypeModel) -> None:
        """
        Updates the current AgentType instance with values from an AgentTypeModel instance.
        """
        # Track if any updates are needed
        updated = False

        if self.description != model.description:
            self.description = model.description
            updated = True

        if self.image != model.image:
            self.image = model.image
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

        # If anything was updated, set the updated timestamp and save changes
        if updated:
            self.updated = time.time()
            self.save()
