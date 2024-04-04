import os
import logging
from typing import Optional, List, Dict
import json

from litellm import Router
from taskara import Task

from .models import EnvVarOptModel, LLMProviderOption


class LLMProvider:
    """
    A chat provider based on LiteLLM
    """

    provider_api_keys: Dict[str, str] = {
        "gpt-4-vision-preview": "OPENAI_API_KEY",
        "anthropic/claude-3-opus-20240229": "ANTHROPIC_API_KEY",
        "gemini/gemini-pro-vision": "GEMINI_API_KEY",
    }

    def __init__(
        self,
        llm_providers: List[str],
        timeout: int = 30,
        allow_fails: int = 1,
        num_retries: int = 3,
    ) -> None:
        self.model_list = []
        fallbacks = []

        if len(llm_providers) == 0:
            raise Exception("No chat providers specified.")

        self.model = llm_providers[0]

        # Construct the model list based on provided preferences and available API keys
        for provider in llm_providers:
            api_key_env = self.provider_api_keys.get(provider)
            if api_key_env:
                api_key = os.getenv(api_key_env)
                if api_key:
                    self.model_list.append(
                        {
                            "model_name": provider,
                            "litellm_params": {
                                "model": provider,
                                "api_key": api_key,
                            },
                        }
                    )

        if len(self.model_list) == 0:
            raise Exception("No valid API keys found for the specified providers.")

        # Calculate fallbacks dynamically
        for i, model in enumerate(self.model_list):
            fallback_models = self.model_list[i + 1 :]
            if fallback_models:
                fallbacks.append(
                    {
                        model["model_name"]: [
                            fallback["model_name"] for fallback in fallback_models
                        ]
                    }
                )

        self.router = Router(
            model_list=self.model_list,
            timeout=timeout,
            allowed_fails=allow_fails,
            num_retries=num_retries,
            set_verbose=True,
            fallbacks=fallbacks,
        )

    @classmethod
    def all_opts(cls) -> List[LLMProviderOption]:
        out = []
        for model, key in cls.provider_api_keys.items():
            out.append(
                LLMProviderOption(
                    model=model,
                    env_var=EnvVarOptModel(
                        name=key,
                        description=f"{model} API key",
                        required=True,
                        secret=True,
                    ),
                )
            )
        return out

    def chat(
        self, msgs: list, task: Optional[Task] = None, namespace: Optional[str] = None
    ) -> dict:
        """Chat with a language model

        Args:
            msgs (list): Messages in openai schema format
            task (Optional[Task], optional): Optional task to log into. Defaults to None.
            namespace (Optional[str], optional): Namespace to log into. Defaults to None.

        Returns:
            dict: The message dictionary
        """

        print(f"calling chat completion for model {self.model} with msgs: ", msgs)
        response = self.router.completion(self.model, msgs)

        print("llm response: ", response.__dict__)
        logging.debug("response: ", response)

        if task:
            dump = {"request": msgs, "response": response.json()}
            if namespace:
                dump["namespace"] = namespace
            print("\ndump: ", dump)
            task.post_message("assistant", json.dumps(dump), thread="prompt")

        return response["choices"][0]["message"].model_dump()

    def check_model(self) -> None:
        """Check if the model is available"""
        msg = {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Just checking if you are working... please return 'yes' if you are",
                },
            ],
        }
        response = self.chat([msg])
        print("response from checking oai functionality: ", response)

    def options(self) -> List[LLMProviderOption]:
        """Dynamically generates options based on the configured providers."""
        options = []
        for model_info in self.model_list:
            model_name = model_info["model_name"]
            api_key_env = self.provider_api_keys.get(model_name)
            if api_key_env:
                option = LLMProviderOption(
                    model=model_name,
                    env_var=EnvVarOptModel(
                        name=api_key_env,
                        description=f"{model_name} API key",
                        required=True,
                        secret=True,
                    ),
                )
                options.append(option)
        return options

    @classmethod
    def from_env(cls):
        """
        Class method to create an LLMProvider instance based on the API keys available in the environment variables.
        """
        available_providers = []

        for provider, env_var in cls.provider_api_keys.items():
            if os.getenv(env_var):
                print(
                    f"\nFound LLM provider '{provider}' API key in environment variables."
                )
                available_providers.append(provider)

        if not available_providers:
            raise ValueError("No API keys found in environment variables.")

        return cls(available_providers)
