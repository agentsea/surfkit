import uuid
import time

from sqlalchemy import Column, String, Boolean, Float, Integer
from sqlalchemy.orm import declarative_base


Base = declarative_base()


class AgentTypeRecord(Base):
    __tablename__ = "agent_types"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, unique=True, index=True)
    description = Column(String)
    image = Column(String)
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
