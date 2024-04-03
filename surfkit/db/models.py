import uuid
import time

from sqlalchemy import Column, String, Boolean, Float, Integer
from sqlalchemy.orm import declarative_base

from ..models import V1UserProfile

Base = declarative_base()


class UserRecord(Base):
    __tablename__ = "users"

    email = Column(String, unique=True, index=True, primary_key=True)
    display_name = Column(String)
    handle = Column(String)
    picture = Column(String)
    created = Column(Integer)
    updated = Column(Integer)

    def to_v1_schema(self) -> V1UserProfile:
        return V1UserProfile(
            email=self.email,
            display_name=self.display_name,
            picture=self.picture,
            created=self.created,
            updated=self.updated,
        )


class AgentTypeRecord(Base):
    __tablename__ = "agent_types"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, unique=True, index=True)
    description = Column(String)
    image = Column(String)
    versions = Column(String)
    env_opts = Column(String)
    supported_runtimes = Column(String)
    owner_id = Column(String)
    public = Column(Boolean)
    icon = Column(String)
    created = Column(Float, default=time.time)
    updated = Column(Float, default=time.time)
    mem_request = Column(String)
    mem_limit = Column(String)
    cpu_request = Column(String)
    cpu_limit = Column(String)
    gpu_mem = Column(String)
    llm_providers = Column(String, nullable=True)
