import time
import uuid

from sqlalchemy import Boolean, Column, Float, Index, Integer, String
from sqlalchemy.orm import declarative_base
from sqlalchemy.inspection import inspect

def to_dict(instance):
    return {
        c.key: getattr(instance, c.key)
        for c in inspect(instance).mapper.column_attrs
    }

Base = declarative_base()


class SkillRecord(Base):
    __tablename__ = "skills"
    __table_args__ = (Index("idx_skill_owner_id", "owner_id"),)
    id = Column(String, primary_key=True)
    owner_id = Column(String, nullable=False)
    name = Column(String, nullable=False)
    status = Column(String, nullable=False)
    description = Column(String, nullable=False)
    requirements = Column(String, nullable=True)
    max_steps = Column(Integer, nullable=False)
    review_requirements = Column(String, nullable=True)
    agent_type = Column(String, nullable=False)
    threads = Column(String, nullable=True)
    generating_tasks = Column(Boolean, default=False, server_default="false")
    example_tasks = Column(String, nullable=True)
    tasks = Column(String, nullable=True)
    public = Column(Boolean, nullable=True, default=False)
    min_demos = Column(Integer, nullable=False)
    demos_outstanding = Column(Integer, nullable=False)
    demo_queue_size = Column(Integer, nullable=False)
    kvs = Column(String, nullable=True)
    created = Column(Float, default=time.time)
    updated = Column(Float, default=time.time)


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
    supports = Column(String)
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
    namespace = Column(String, nullable=True)


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
    icon = Column(String, nullable=True)
    owner_id = Column(String, nullable=True)
    created = Column(Float, default=time.time)
    updated = Column(Float, default=time.time)
