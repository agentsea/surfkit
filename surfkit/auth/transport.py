import logging
import os
from typing import Annotated

from agentcore.models import V1UserProfile
from fastapi import Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer

from .provider import default_auth_provider

logger = logging.getLogger(__name__)

if os.getenv("AGENT_NO_AUTH", "false").lower() == "true":
    user_auth = None
else:
    user_auth = default_auth_provider()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


async def get_current_user(
    token: Annotated[str, Depends(oauth2_scheme)],
) -> V1UserProfile:
    if not user_auth:
        raise SystemError("user auth is not configured")
    try:
        print(f"checking user token: {token}", flush=True)
        user = user_auth.get_user_auth(token)
        user.token = token
    except Exception as e:
        logging.error(e)
        raise HTTPException(
            status_code=401,
            detail=f"-ID token was unauthorized, please log in: {e}",
        )

    return user


async def get_user_mock_auth() -> V1UserProfile:
    # Return a dummy user profile when authentication is disabled
    return V1UserProfile(
        email="tom@myspace.com",
        display_name="tom",
        picture="https://i.insider.com/4efd9b8b69bedd682c000022?width=750&format=jpeg&auto=webp",
    )


def get_user_dependency():
    if os.getenv("AGENT_NO_AUTH", "false").lower() == "true":
        print("using mock auth", flush=True)
        return get_user_mock_auth
    else:
        print("using current user", flush=True)
        return get_current_user


def get_current_user_sync(
    token: str,
) -> V1UserProfile:
    if not user_auth:
        raise SystemError("user auth is not configured")
    try:
        print(f"checking user token: {token}", flush=True)
        user = user_auth.get_user_auth(token)
        user.token = token

    except Exception as e:
        logging.error(e)
        raise HTTPException(
            status_code=401,
            detail=f"-ID token was unauthorized, please log in: {e}",
        )

    return user


def get_user_mock_auth_sync() -> V1UserProfile:
    # Return a dummy user profile when authentication is disabled
    return V1UserProfile(
        email="tom@myspace.com",
        display_name="tom",
        picture="https://i.insider.com/4efd9b8b69bedd682c000022?width=750&format=jpeg&auto=webp",
    )


def get_user_dependency_sync():
    if os.getenv("AGENT_NO_AUTH", "false").lower() == "true":
        print("using mock auth", flush=True)
        return get_user_mock_auth_sync
    else:
        print("using current user", flush=True)
        return get_current_user_sync
