import os

import requests

from .config import AGENTSEA_AUTH_URL
from .server.models import V1UserProfile


class HubAuth:
    """The Agentsea Hub Auth"""

    def __init__(self, hub_auth_url: str = AGENTSEA_AUTH_URL) -> None:
        self.hub_auth_url = hub_auth_url

    def get_api_key(self, token: str) -> str:
        """Get the api key from the hub"""

        hub_key_url = f"{self.hub_auth_url}/v1/users/me/keys"
        headers = {"Authorization": f"Bearer {token}"}
        response = requests.get(hub_key_url, headers=headers)
        response.raise_for_status()
        key_data = response.json()

        return key_data["keys"][0]["key"]

    def get_user_info(self, token: str) -> V1UserProfile:
        """Get user info from the hub"""

        hub_user_url = f"{self.hub_auth_url}/v1/users/me"
        headers = {"Authorization": f"Bearer {token}"}
        response = requests.get(hub_user_url, headers=headers)
        response.raise_for_status()
        user_data = response.json()

        return V1UserProfile.model_validate(user_data)
