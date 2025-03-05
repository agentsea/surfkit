from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from taskara import V1Task
from taskara.review import V1ReviewRequirement
from threadmem import V1RoleThread


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
    supports: List[str] = []
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
    namespace: Optional[str] = None


class V1AgentTypes(BaseModel):
    types: List[V1AgentType]


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
    icon: Optional[str] = None
    created: float
    updated: float


class V1AgentInstances(BaseModel):
    instances: List[V1AgentInstance]


class V1Find(BaseModel):
    args: dict = {}


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


class UserTasks(BaseModel):
    """A list of tasks for a user story"""

    tasks: List[str] = Field(description="A list of tasks for a user story")


class UserTask(BaseModel):
    """A task for a user story"""

    task: str = Field(description="A task for a user story")


class V1Skill(BaseModel):
    id: str
    name: str
    description: str
    requirements: List[str]
    max_steps: int
    review_requirements: List[V1ReviewRequirement]
    tasks: List[V1Task]
    example_tasks: List[str]
    threads: List[V1RoleThread] = []
    status: Optional[str] = None
    min_demos: Optional[int] = None
    demos_outstanding: Optional[int] = None
    demo_queue_size: Optional[int] = None
    owner_id: Optional[str] = None
    generating_tasks: Optional[bool] = None
    agent_type: str
    kvs: Optional[Dict[str, Any]] = None
    remote: Optional[str] = None
    created: int
    updated: int


class SkillsWithGenTasks(BaseModel):
    skill_id: str
    in_queue_count: int
    tasks_needed: int


class V1UpdateSkill(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    requirements: Optional[List[str]] = None
    max_steps: Optional[int] = None
    review_requirements: Optional[List[V1ReviewRequirement]] = None
    tasks: Optional[List[str]] = None
    example_tasks: Optional[List[str]] = None
    threads: Optional[List[str]] = None
    status: Optional[str] = None
    min_demos: Optional[int] = None
    demos_outstanding: Optional[int] = None
    demo_queue_size: Optional[int] = None
    kvs: Optional[Dict[str, Any]] = None


class V1SetKey(BaseModel):
    key: str
    value: str


class V1LearnSkill(BaseModel):
    skill_id: str
    remote: Optional[str] = None
    agent: Optional[V1Agent] = None


class V1LearnTask(BaseModel):
    task: V1Task
    remote: Optional[str] = None
    agent: Optional[V1Agent] = None
