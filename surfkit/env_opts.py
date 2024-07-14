import os
from typing import Optional

import typer
import openai
from mllm import Router

from surfkit.types import AgentType


def find_local_llm_keys(typ: AgentType) -> Optional[dict]:

    env_vars = None

    if typ.llm_providers and typ.llm_providers.preference:
        found = {}
        for provider_name in typ.llm_providers.preference:
            api_key_env = Router.provider_api_keys.get(provider_name)
            if not api_key_env:
                raise ValueError(f"no api key env for provider {provider_name}")
            key = os.getenv(api_key_env)
            if not key:
                print("no api key found locally for provider: ", provider_name)
                continue

            typer.echo(f"Added local API key for {provider_name}")
            found[api_key_env] = key

        if not found:
            raise ValueError(
                "no api keys found locally for any of the providers in the agent type"
            )
        env_vars = found

    return env_vars


def is_api_key_valid(api_key: str) -> bool:
    client = openai.OpenAI(api_key=api_key)
    try:
        client.models.list()
    except openai.AuthenticationError:
        return False
    else:
        return True


def find_llm_keys(typ: AgentType, llm_providers_local: bool) -> Optional[dict]:
    env_vars = None
    if typ.llm_providers and typ.llm_providers.preference:
        found = {}

        if llm_providers_local:
            found = find_local_llm_keys(typ)

            if found is None:
                found = {}
            else:
                env_vars = found

        if not found:
            typer.echo("\nThis agent requires one of the following LLM API keys:")
            for provider_name in typ.llm_providers.preference:
                api_key_env = Router.provider_api_keys.get(provider_name)
                if api_key_env:
                    typer.echo(f"   - {api_key_env}")
            typer.echo("")

            for provider_name in typ.llm_providers.preference:
                api_key_env = Router.provider_api_keys.get(provider_name)
                if not api_key_env:
                    raise ValueError(f"No API key env for provider {provider_name}")

                if found.get(api_key_env):
                    continue

                key = os.getenv(api_key_env)
                if not key:
                    continue

                add = typer.confirm(
                    f"Would you like to add your local API key for '{provider_name}'"
                )
                if add:
                    found[api_key_env] = key

            if not found:
                for provider_name in typ.llm_providers.preference:
                    while True:
                        add = typer.confirm(
                            f"Would you like to enter an API key for '{provider_name}'"
                        )
                        if add:
                            api_key_env = Router.provider_api_keys.get(provider_name)
                            if not api_key_env:
                                continue
                            response = typer.prompt(api_key_env)
                            # TODO: validate other providers
                            if api_key_env != "OPENAI_API_KEY":
                                found[api_key_env] = response
                                break
                            if is_api_key_valid(response):
                                found[api_key_env] = response
                                break
                            else:
                                typer.echo(
                                    f"The API Key is not valid for '{provider_name}'. Please try again."
                                )
                        else:
                            break

            if not found:
                raise ValueError(
                    "No valid API keys given for any of the llm providers in the agent type"
                )

            env_vars = found

    return env_vars


def find_env_opts(typ: AgentType, use_local: bool) -> Optional[dict]:
    env_vars = None

    for env_opt in typ.env_opts:
        found = {}
        key = os.getenv(env_opt.name)
        if key:
            if use_local:
                found[env_opt.name] = key
                typer.echo(f"Added local API key for {env_opt.name}")
            else:
                add = typer.confirm(
                    f"Would you like to add your local API key for '{env_opt.name}'"
                )
                if add:
                    found[env_opt.name] = key

        if not found:
            if not env_opt.required:
                add = typer.confirm(
                    f"Would you like to enter an API key for '{env_opt.name}'"
                )
                if add:
                    response = typer.prompt(f"{env_opt.name}")
                    found[env_opt.name] = response
            else:
                response = typer.prompt(f"Please enter {env_opt.name}")
                found[env_opt.name] = response

        if found:
            env_vars = found

    return env_vars


def find_envs(typ: AgentType, use_local: bool) -> Optional[dict]:
    env_vars = None
    llm_envs = find_llm_keys(typ, use_local)
    if llm_envs:
        env_vars = llm_envs
    envs = find_env_opts(typ, use_local)
    if envs:
        if env_vars:
            env_vars.update(envs)
        else:
            env_vars = envs

    return env_vars
