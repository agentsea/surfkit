import sys
import traceback
import warnings
from typing import Any, Dict, List, Optional

from pydantic import BaseModel
from taskara import V1Task


class V1Action(BaseModel):
    """An action"""

    name: str
    parameters: Dict[str, Any]


class V1ActionSelection(BaseModel):
    """An action selection from the model"""

    observation: str
    reason: str
    action: V1Action


class V1DeviceConfig(BaseModel):
    name: str
    provision: bool = False


class V1DevicesConfig(BaseModel):
    preference: List[V1DeviceConfig] = []


class V1Runtime(BaseModel):
    type: str
    preference: List[str] = []


class V1ResourceLimits(BaseModel):
    cpu: str = "2"
    memory: str = "2Gi"


class V1ResourceRequests(BaseModel):
    cpu: str = "1"
    memory: str = "500m"
    gpu: Optional[str] = None


class V1EnvVarOpt(BaseModel):
    name: str
    description: Optional[str] = None
    required: bool = False
    default: Optional[str] = None
    secret: bool = False
    options: List[str] = []


class V1LLMProviders(BaseModel):
    preference: List[str] = []


class V1Agent(BaseModel):
    name: str
    config: Dict[str, Any]


class V1SolveTask(BaseModel):
    task: V1Task
    agent: Optional[V1Agent] = None


class V1CreateTask(BaseModel):
    task: V1Task
    agent: Optional[V1Agent] = None


class V1Meter(BaseModel):
    name: str
    unit: str
    cost: float
    description: Optional[str] = None


class V1RuntimeConnect(BaseModel):
    name: str
    connect_config: BaseModel


class V1AgentType(BaseModel):
    version: Optional[str] = None
    kind: Optional[str] = None
    id: Optional[str] = None
    name: str
    description: str
    cmd: str
    owner_id: Optional[str] = None
    repo: Optional[str] = None
    img_repo: Optional[str] = None
    versions: Optional[Dict[str, str]] = None
    env_opts: List[V1EnvVarOpt] = []
    runtimes: List[V1Runtime] = []
    created: Optional[float] = None
    updated: Optional[float] = None
    public: bool = False
    icon: Optional[str] = None
    resource_requests: V1ResourceRequests = V1ResourceRequests()
    resource_limits: V1ResourceLimits = V1ResourceLimits()
    llm_providers: Optional[V1LLMProviders] = None
    devices: List[V1DeviceConfig] = []
    meters: List[V1Meter] = []
    tags: List[str] = []
    labels: Dict[str, str] = {}


class V1AgentInstance(BaseModel):
    name: str
    type: V1AgentType
    runtime: V1RuntimeConnect
    version: Optional[str] = None
    port: int = 9090
    labels: Dict[str, str] = {}
    tags: List[str] = []
    status: str
    owner_id: Optional[str] = None
    created: float
    updated: float


class V1Find(BaseModel):
    args: dict = {}


class V1AgentTypes(BaseModel):
    types: List[V1AgentType]


class V1CreateAgentType(BaseModel):
    id: str
    name: str
    description: str
    image: str
    env_opts: List[V1EnvVarOpt] = []
    supported_runtimes: List[str] = []
    versions: Dict[str, str] = {}
    public: bool = False
    icon: Optional[str] = None
    tags: List[str] = []
    labels: Dict[str, str] = {}


class V1Work:
    remote: str
    check_interval: int


class V1UserProfile(BaseModel):
    email: Optional[str] = None
    display_name: Optional[str] = None
    handle: Optional[str] = None
    picture: Optional[str] = None
    created: Optional[int] = None
    updated: Optional[int] = None
    token: Optional[str] = None


class V1Meta(BaseModel):
    id: str
    tags: List[str] = []
    labels: Dict[str, str] = {}
    owner_id: Optional[str] = None
    created: float
    updated: float
