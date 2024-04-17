from typing import List, Optional, Dict

from pydantic import BaseModel


class EnvVarOptModel(BaseModel):
    name: str
    description: Optional[str] = None
    required: bool = False
    default: Optional[str] = None
    secret: bool = False
    options: List[str] = []


class LLMProviders(BaseModel):
    preference: List[str] = []


class DeviceConfig(BaseModel):
    name: str
    provision: bool = False


class DevicesConfig(BaseModel):
    preference: List[DeviceConfig] = []


class LLMProviderOption(BaseModel):
    model: str
    env_var: EnvVarOptModel


class LLMProviderModel(BaseModel):
    options: List[LLMProviderOption]


class MeterModel(BaseModel):
    name: str
    unit: str
    cost: float
    description: Optional[str] = None


class RuntimeModel(BaseModel):
    type: str
    preference: List[str] = []


class ResourceLimitsModel(BaseModel):
    cpu: str = "2"
    memory: str = "2Gi"


class ResourceRequestsModel(BaseModel):
    cpu: str = "1"
    memory: str = "500m"
    gpu: Optional[str] = None


class AgentTypeModel(BaseModel):
    version: Optional[str] = None
    kind: Optional[str] = None
    id: Optional[str] = None
    name: str
    description: str
    owner_id: Optional[str] = None
    repo: Optional[str] = None
    image: Optional[str] = None
    versions: Optional[Dict[str, str]] = None
    env_opts: List[EnvVarOptModel] = []
    runtimes: List[RuntimeModel] = []
    created: Optional[float] = None
    updated: Optional[float] = None
    public: bool = False
    icon: Optional[str] = None
    resource_requests: ResourceRequestsModel = ResourceRequestsModel()
    resource_limits: ResourceLimitsModel = ResourceLimitsModel()
    llm_providers: Optional[LLMProviders] = None
    devices: List[DeviceConfig] = []
    meters: List[MeterModel] = []


class AgentTypesModel(BaseModel):
    types: List[AgentTypeModel]


class CreateAgentTypeModel(BaseModel):
    id: str
    name: str
    description: str
    image: str
    env_opts: List[EnvVarOptModel] = []
    supported_runtimes: List[str] = []
    versions: Dict[str, str] = {}
    public: bool = False
    icon: Optional[str] = None


class V1UserProfile(BaseModel):
    email: Optional[str] = None
    display_name: Optional[str] = None
    handle: Optional[str] = None
    picture: Optional[str] = None
    created: Optional[int] = None
    updated: Optional[int] = None
    token: Optional[str] = None
