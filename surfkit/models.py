from typing import List, Optional, Dict

from pydantic import BaseModel
from taskara.models import TaskModel


class EnvVarOptModel(BaseModel):
    name: str
    description: Optional[str] = None
    required: bool = False
    default: Optional[str] = None
    secret: bool = False
    options: List[str] = []


class LLMProviders(BaseModel):
    preference: List[str] = []


class LLMProviderOption(BaseModel):
    model: str
    env_var: EnvVarOptModel


class LLMProviderModel(BaseModel):
    options: List[LLMProviderOption]


class AgentTypeModel(BaseModel):
    id: str
    name: str
    owner_id: str
    description: str
    image: str
    versions: Dict[str, str]
    env_opts: List[EnvVarOptModel] = []
    supported_runtimes: List[str] = []
    created: float
    updated: float
    public: bool = False
    icon: Optional[str] = None
    mem_request: Optional[str] = "500m"
    mem_limit: Optional[str] = "2gi"
    cpu_request: Optional[str] = "1"
    cpu_limit: Optional[str] = "4"
    gpu_mem: Optional[str] = None
    llm_providers: Optional[LLMProviders] = None


class AgentTypesModel(BaseModel):
    types: List[AgentTypeModel]


class CreateAgentTypeModel(BaseModel):
    id: str
    name: str
    description: str
    image: str
    env_opts: List[EnvVarOptModel] = []
    supported_runtimes: List[str] = []
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
