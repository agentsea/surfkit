import uuid
import time

from sqlalchemy import Column, String, ForeignKey, Boolean, Float, Integer
from sqlalchemy.orm import relationship, declarative_base

from guisurfer.server.models import V1UserProfile

Base = declarative_base()


class SSHKeyRecord(Base):
    __tablename__ = "ssh_keys"

    id = Column(String, primary_key=True, index=True)
    owner_id = Column(String, nullable=False)
    public_key = Column(String, unique=True, index=True)
    private_key = Column(String)
    name = Column(String, index=True)
    created = Column(Float)
    full_name = Column(String, unique=True, index=True)
    metadata_ = Column(String)


class SharedDesktopRuntimeRecord(Base):
    __tablename__ = "shared_desktop_runtimes"

    id = Column(Integer, primary_key=True, autoincrement=True)
    runtime_id = Column(String, ForeignKey("desktop_runtimes.id"), nullable=False)
    shared_with_user_id = Column(String, nullable=False)
    # Additional fields like share_date can be added here
    runtime = relationship("DesktopRuntimeRecord", back_populates="shared_with")


class DesktopRuntimeRecord(Base):
    __tablename__ = "desktop_runtimes"

    id = Column(String, primary_key=True, index=True)
    owner_id = Column(String, nullable=False)
    name = Column(String, index=True)
    provider = Column(String, index=True)
    credentials = Column(String)
    created = Column(Float)
    updated = Column(Float)
    full_name = Column(String, unique=True, index=True)
    metadata_ = Column(String)
    shared_with = relationship("SharedDesktopRuntimeRecord", back_populates="runtime")


class SharedAgentRuntimeRecord(Base):
    __tablename__ = "shared_agent_runtimes"

    id = Column(Integer, primary_key=True, autoincrement=True)
    runtime_id = Column(String, ForeignKey("agent_runtimes.id"), nullable=False)
    shared_with_user_id = Column(String, nullable=False)
    runtime = relationship("AgentRuntimeRecord", back_populates="shared_with")


class AgentRuntimeRecord(Base):
    __tablename__ = "agent_runtimes"

    id = Column(String, primary_key=True, index=True)
    owner_id = Column(String, nullable=False)
    name = Column(String, index=True)
    provider = Column(String, index=True)
    credentials = Column(String)
    created = Column(Float)
    updated = Column(Float)
    full_name = Column(String, unique=True, index=True)
    metadata_ = Column(String)
    shared_with = relationship("SharedAgentRuntimeRecord", back_populates="runtime")


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


class AgentRecord(Base):
    __tablename__ = "agents"

    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    owner_id = Column(String, nullable=False)
    type = Column(String, nullable=False)
    runtime = Column(String, nullable=False)
    status = Column(String, nullable=False)
    secrets = Column(String, nullable=True)
    desktop = Column(String, nullable=True)
    create_job_id = Column(String, nullable=True)
    created = Column(Float, nullable=False)
    updated = Column(Float, nullable=False)
    metadata_ = Column(String, nullable=True)
    envs = Column(String, nullable=True)
    icon = Column(String, nullable=True)
    full_name = Column(String, unique=True, index=True)


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


class JobRecord(Base):
    __tablename__ = "jobs"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    owner_id = Column(String, nullable=False)
    name = Column(String, nullable=False)
    type = Column(String, nullable=False)
    status = Column(String, nullable=False)
    runtime = Column(String, nullable=False)
    namespace = Column(String, nullable=True)
    phase = Column(String, nullable=True)
    logs = Column(String, nullable=True)
    result = Column(String, nullable=True)
    created = Column(Float, default=time.time)
    updated = Column(Float, default=time.time)
    finished = Column(Float, default=0.0)
    metadata_ = Column(String, nullable=True)
