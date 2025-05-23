from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional

# Default provider configurations
PROVIDERS = {
    "openai": {
        "name": "OpenAI",
        "base_url": "https://api.openai.com/v1",
        "env_key": "OPENAI_API_KEY",
    },
    "openrouter": {
        "name": "OpenRouter",
        "base_url": "https://openrouter.ai/api/v1",
        "env_key": "OPENROUTER_API_KEY",
    },
    "azure": {
        "name": "Azure OpenAI",
        "base_url": None,  # Must be provided by user
        "env_key": "AZURE_OPENAI_API_KEY",
    },
    "gemini": {
        "name": "Google Gemini",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "env_key": "GEMINI_API_KEY",
    },
    "ollama": {
        "name": "Ollama",
        "base_url": "http://localhost:11434/v1",
        "env_key": "OLLAMA_API_KEY",
    },
    "mistral": {
        "name": "Mistral AI",
        "base_url": "https://api.mistral.ai/v1",
        "env_key": "MISTRAL_API_KEY",
    },
    "deepseek": {
        "name": "DeepSeek",
        "base_url": "https://api.deepseek.com",
        "env_key": "DEEPSEEK_API_KEY",
    },
    "xai": {"name": "xAI", "base_url": "https://api.x.ai/v1", "env_key": "XAI_API_KEY"},
    "groq": {
        "name": "Groq",
        "base_url": "https://api.groq.com/openai/v1",
        "env_key": "GROQ_API_KEY",
    },
    "arceeai": {
        "name": "ArceeAI",
        "base_url": "https://conductor.arcee.ai/v1",
        "env_key": "ARCEEAI_API_KEY",
    },
}


@dataclass
class ProviderConfig:
    name: str
    base_url: Optional[str]
    env_key: str

    @classmethod
    def get_provider(cls, provider_name: str) -> ProviderConfig:
        """Get provider configuration by name"""
        if provider_name not in PROVIDERS:
            # For custom providers, use the name as the env key prefix
            return ProviderConfig(
                name=provider_name.capitalize(),
                base_url=os.environ.get(f"{provider_name.upper()}_BASE_URL"),
                env_key=f"{provider_name.upper()}_API_KEY",
            )

        config = PROVIDERS[provider_name]
        return ProviderConfig(
            name=config["name"], base_url=config["base_url"], env_key=config["env_key"]
        )

    def get_api_key(self) -> Optional[str]:
        """Get the API key for this provider from environment variables"""
        return os.environ.get(self.env_key)
