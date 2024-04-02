from typing import List, Optional

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


class LLMProviderOption(BaseModel):
    model: str
    env_var: EnvVarOptModel


class LLMProviderModel(BaseModel):
    options: List[LLMProviderOption]
