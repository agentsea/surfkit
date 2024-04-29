import os
import logging
from typing import Optional, List, Dict, TypeVar, Type, Generic
import json
import time

from litellm import Router, ModelResponse
from taskara import Task, Prompt
from threadmem import RoleThread, RoleMessage
from litellm._logging import handler
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt

from .models import EnvVarOptModel, LLMProviderOption
from .types import AgentType
from .util import extract_parse_json


T = TypeVar("T", bound=BaseModel)


class ChatResponse(Generic[T], BaseModel):
    model: str
    msg: RoleMessage
    parsed: Optional[T] = None
    time_elapsed: float
    tokens_request: int
    tokens_response: int


class LLMProvider:
    """
    A chat provider based on LiteLLM
    """

    provider_api_keys: Dict[str, str] = {
        "gpt-4-turbo": "OPENAI_API_KEY",
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
            set_verbose=False,
            debug_level="INFO",
            fallbacks=fallbacks,
        )

        verbose_router_logger = logging.getLogger("LiteLLM Router")
        verbose_router_logger.setLevel(logging.ERROR)
        verbose_logger = logging.getLogger("LiteLLM")
        verbose_logger.setLevel(logging.ERROR)
        handler.setLevel(logging.ERROR)

    @classmethod
    def opts_for_type(cls, type: AgentType) -> List[LLMProviderOption]:
        out = []
        for model, key in cls.provider_api_keys.items():
            if type.llm_providers and model in type.llm_providers.preference:
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
        self,
        thread: RoleThread,
        model: Optional[str] = None,
        task: Optional[Task] = None,
        namespace: str = "default",
        response_schema: Optional[Type[T]] = None,
        retries: int = 3,
    ) -> ChatResponse[T]:
        """Chat with a language model

        Args:
            thread (RoleThread): A role thread
            model (Optional[str], optional): Model to use. Defaults to None.
            task (Optional[Task], optional): Optional task to log into. Defaults to None.
            namespace (Optional[str], optional): Namespace to log into. Defaults to "default".
            response_schema (Optional[Type[T]], optional): Schema to validate response against. Defaults to None.
            retries (int, optional): Number of retries if model fails. Defaults to 3.

        Returns:
            ChatResponse: A chat response
        """
        if not model:
            model = self.model

        @retry(stop=stop_after_attempt(retries))
        def call_llm(
            thread: RoleThread,
            model: str,
            task: Optional[Task] = None,
            namespace: str = "default",
            response_schema: Optional[Type[T]] = None,
        ) -> ChatResponse[T]:
            start = time.time()
            response = self.router.completion(model, thread.to_openai())

            if not isinstance(response, ModelResponse):
                raise Exception(f"Unexpected response type: {type(response)}")

            end = time.time()

            elapsed = end - start

            print("llm response: ", response.__dict__)
            logging.debug("response: ", response)

            response_obj = None
            msg = response["choices"][0]["message"].model_dump()
            if response_schema:
                try:
                    # type: ignore
                    response_obj = response_schema.model_validate(
                        extract_parse_json(msg["text"])
                    )
                except Exception as e:
                    print("Validation error: ", e)
                    raise

            resp_msg = RoleMessage(role=msg["role"], text=msg["text"])
            out = ChatResponse(
                model=response.model or model,
                msg=resp_msg,
                parsed=response_obj,
                time_elapsed=elapsed,
                tokens_request=0,  # TODO
                tokens_response=0,
            )
            if task:
                task.store_prompt(thread, resp_msg, namespace)

            return out

        return call_llm(thread, model, task, namespace, response_schema)

    def check_model(self) -> None:
        """Check if the model is available"""

        thread = RoleThread()
        thread.post(
            "user", "Just checking if you are working... please return 'yes' if you are"
        )
        response = self.chat(thread)
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

        preference_data = os.getenv("MODEL_PREFERENCE")
        preference = None
        if preference_data:
            preference = preference_data.split(",")
        if not preference:
            preference = cls.provider_api_keys.keys()

        print("\nloading models with preference: ", preference)
        for provider in preference:
            env_var = cls.provider_api_keys.get(provider)
            if not env_var:
                raise ValueError(
                    f"Invalid provider '{provider}' specified in MODEL_PREFERENCE."
                )
            if os.getenv(env_var):
                print(
                    f"\nFound LLM provider '{provider}' API key in environment variables."
                )
                available_providers.append(provider)

        if not available_providers:
            raise ValueError("No API keys found in environment variables.")

        return cls(available_providers)
