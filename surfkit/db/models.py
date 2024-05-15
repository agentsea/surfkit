import time
import uuid

from sqlalchemy import Boolean, Column, Float, Integer, String
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class AgentTypeRecord(Base):
    __tablename__ = "agent_types"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, unique=True, index=True)
    description = Column(String)
    kind = Column(String)
    cmd = Column(String)
    img_repo = Column(String)
    versions = Column(String, nullable=True)
    repo = Column(String, nullable=True)
    env_opts = Column(String)
    runtimes = Column(String)
    owner_id = Column(String)
    public = Column(Boolean)
    icon = Column(String)
    created = Column(Float, default=time.time)
    updated = Column(Float, default=time.time)
    resource_requests = Column(String, nullable=True)
    resource_limits = Column(String, nullable=True)
    llm_providers = Column(String, nullable=True)
    devices = Column(String, nullable=True)
    meters = Column(String, nullable=True)
    tags = Column(String, nullable=True)
    labels = Column(String, nullable=True)


class AgentStatusRecord(Base):
    __tablename__ = "agent_status"

    agent_id = Column(String, primary_key=True)
    status = Column(String)
    task_id = Column(String, nullable=True)


class AgentInstanceRecord(Base):
    __tablename__ = "agent_instances"

    id = Column(String, primary_key=True)
    name = Column(String, unique=True, index=True)
    full_name = Column(String)
    type = Column(String)
    runtime_name = Column(String)
    runtime_config = Column(String)
    version = Column(String, nullable=True)
    status = Column(String)
    tags = Column(String, nullable=True)
    labels = Column(String, nullable=True)
    port = Column(Integer)
    owner_id = Column(String, nullable=True)
    created = Column(Float, default=time.time)
    updated = Column(Float, default=time.time)
